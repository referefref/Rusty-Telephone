use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use anyhow::{Result, Context};
use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rustfft::{FftPlanner, num_complex::Complex};
use sha2::{Sha256, Digest};
use ringbuf::HeapRb;
use indicatif::{ProgressBar, ProgressStyle, MultiProgress};
use console::style;

const SAMPLE_RATE: u32 = 48000;
const BIT_DURATION_MS: u64 = 2;
const SYNC_FREQUENCY: f32 = 240.0;
const SYNC_DURATION_MS: u64 = 1000;
const PREAMBLE_FREQUENCY: f32 = 600.0;
const PREAMBLE_DURATION_MS: u64 = 500;
const FREQUENCY_TOLERANCE: f32 = 100.0;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = ".")]
    output_dir: PathBuf,
}

#[derive(Clone, Copy, Debug)]
struct FrequencyMap {
    zero: f32,
    one: f32,
    start: f32,
    end: f32,
}

impl Default for FrequencyMap {
    fn default() -> Self {
        Self {
            zero: 2000.0,
            one: 4000.0,
            start: 1000.0,
            end: 6000.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct FrequencyDetection {
    frequency: f32,
    amplitude: f32,
    rms: f32,
}

#[derive(Debug)]
enum DecodeState {
    WaitingForSync,
    WaitingForPreamble,
    WaitingForByteStart,
    ReadingBits { current_byte: u8, bits_read: usize },
    WaitingForByteEnd,
}

struct ReceiveStats {
    total_samples: usize,
    valid_chunks: usize,
    start_time: std::time::Instant,
    frequencies_detected: Vec<(usize, FrequencyDetection)>,
}

impl ReceiveStats {
    fn new() -> Self {
        Self {
            total_samples: 0,
            valid_chunks: 0,
            start_time: std::time::Instant::now(),
            frequencies_detected: Vec::new(),
        }
    }

    fn print_summary(&self) {
        let duration = self.start_time.elapsed();
        
        println!("\n{}", style("Reception Summary:").cyan().bold());
        println!("├─ Total samples: {}", style(self.total_samples).green());
        println!("├─ Valid chunks: {}", style(self.valid_chunks).green());
        println!("├─ Duration: {:.2}s", style(duration.as_secs_f64()).green());

        println!("\n{}", style("Recent frequency detections:").cyan());
        for (chunk, detection) in self.frequencies_detected.iter().rev().take(10).rev() {
            println!("Chunk {}: {:.1} Hz (RMS: {:.6}, Amp: {:.6})", 
                    chunk, detection.frequency, detection.rms, detection.amplitude);
        }
    }
}

struct AudioDecoder {
    sample_rate: u32,
    frequencies: FrequencyMap,
    fft_planner: FftPlanner<f32>,
    stats: ReceiveStats,
    multi_progress: MultiProgress,
}

impl AudioDecoder {
    fn new(sample_rate: u32) -> Self {
        let frequencies = FrequencyMap::default();
        
        println!("\n{}", style("Frequency configuration:").cyan());
        println!("├─ Start: {:.1} Hz", frequencies.start);
        println!("├─ Zero: {:.1} Hz", frequencies.zero);
        println!("├─ One: {:.1} Hz", frequencies.one);
        println!("└─ End: {:.1} Hz", frequencies.end);
        
        Self {
            sample_rate,
            frequencies,
            fft_planner: FftPlanner::new(),
            stats: ReceiveStats::new(),
            multi_progress: MultiProgress::new(),
        }
    }

    fn detect_frequency(&mut self, samples: &[f32], chunk_index: usize) -> Option<f32> {
        if samples.len() < 96 {
            return None;
        }

        let rms = (samples.iter().map(|x| x * x).sum::<f32>() / samples.len() as f32).sqrt();
        // println!("RMS at chunk {}: {:.6}", chunk_index, rms);
         
        if rms < 0.01 {  // Lowered threshold for better sensitivity
            return None;
        }

        let windowed_samples: Vec<Complex<f32>> = samples.iter()
            .enumerate()
            .map(|(i, &x)| {
                let window = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 
                                         / (samples.len() - 1) as f32).cos());
                Complex::new(x * window, 0.0)
            })
            .collect();

        let mut spectrum = windowed_samples;
        let fft = self.fft_planner.plan_fft_forward(samples.len());
        fft.process(&mut spectrum);

        let bin_size = 5.0; // Reduced bin size for better frequency resolution
        let mut binned_spectrum = vec![0.0f32; (5000.0 / bin_size) as usize];
        
        for (i, c) in spectrum.iter().enumerate() {
            let freq = i as f32 * self.sample_rate as f32 / samples.len() as f32;
            if freq <= 5000.0 {
                let bin = (freq / bin_size) as usize;
                if bin < binned_spectrum.len() {
                    binned_spectrum[bin] += c.norm_sqr() as f32;
                }
            }
        }

        let mean_power = binned_spectrum.iter()
            .filter(|&&p| p > 0.0)
            .sum::<f32>() / binned_spectrum.iter().filter(|&&p| p > 0.0).count() as f32;
        let threshold = mean_power * 1.5; // Lowered threshold multiplier

        let max_freq = binned_spectrum.iter()
            .enumerate()
            .filter(|(_, &power)| power > threshold)
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, power)| {
                let freq = (i as f32 * bin_size) + (bin_size / 2.0);
                
                if chunk_index % 100 == 0 {
                    println!("Chunk {}: Detected frequency {:.1} Hz (power: {:.6})", 
                            chunk_index, freq, power);
                }

                freq
            });

        if max_freq.is_some() {
            self.stats.valid_chunks += 1;
        }

        max_freq
    }

    fn wait_for_sync(&mut self, samples: &[f32]) -> Option<usize> {
        self.detect_signal(samples, SYNC_FREQUENCY, SYNC_DURATION_MS, "sync")
    }

    fn wait_for_preamble(&mut self, samples: &[f32], start_index: usize) -> Option<usize> {
        let window_size = (self.sample_rate as f64 * (PREAMBLE_DURATION_MS as f64 / 1000.0)) as usize;
        let step_size = window_size / 4; // Smaller steps for more precise detection
        
        println!("\nPreamble detection parameters:");
        println!("├─ Start index: {}", start_index);
        println!("├─ Window size: {}", window_size);
        println!("├─ Step size: {}", step_size);
        println!("└─ Available samples: {}", samples.len());
    
        let spinner = self.multi_progress.add(ProgressBar::new_spinner());
        spinner.set_style(ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .expect("Invalid spinner template"));
    
        // Skip past the sync signal
        let search_start = start_index + (SYNC_DURATION_MS as f32 * SAMPLE_RATE as f32 / 1000.0) as usize;
        
        for (i, window) in samples[search_start..].windows(window_size).step_by(step_size).enumerate() {
            spinner.set_message(format!("Looking for preamble... (window {})", i));
            
            if let Some(freq) = self.detect_frequency(window, i) {
                println!("Window {}: Found frequency {:.1} Hz", i, freq);
                
                if (freq - PREAMBLE_FREQUENCY).abs() < FREQUENCY_TOLERANCE {
                    let absolute_position = search_start + (i * step_size);
                    spinner.finish_with_message(format!(
                        "Preamble detected at sample {} (freq: {:.1} Hz)",
                        absolute_position, freq
                    ));
                    return Some(absolute_position + window_size); // Return position after preamble
                }
            }
            
            if i % 10 == 0 {
                println!("Processed {} windows", i);
            }
        }
    
        spinner.finish_with_message("No preamble detected");
        None
    }

    fn detect_signal(&mut self, samples: &[f32], target_frequency: f32, duration_ms: u64, signal_name: &str) -> Option<usize> {
        let samples_per_ms = self.sample_rate as usize / 1000;
        let window_size = duration_ms as usize * samples_per_ms;
        let step_size = samples_per_ms * 30;  // Check every 25ms for more frequent checks
        
        println!("\n{} {} detection parameters:", style(signal_name).cyan(), style("signal").cyan());
        println!("├─ Frequency: {} Hz", target_frequency);
        println!("├─ Duration: {} ms", duration_ms);
        println!("├─ Window size: {} samples", window_size);
        println!("└─ Tolerance: ±{} Hz", FREQUENCY_TOLERANCE);
        println!("├─ Total samples available: {}", samples.len());
        println!("└─ Maximum windows possible: {}", samples.len().saturating_sub(window_size) / step_size + 1);

        let spinner = self.multi_progress.add(ProgressBar::new_spinner());
        spinner.set_style(ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .expect("Invalid spinner template"));

        let max_window_idx = samples.len().saturating_sub(window_size) / step_size;

        println!("Beginning {} signal detection with {} possible windows", signal_name, max_window_idx);

        for (i, window) in samples.windows(window_size).enumerate().step_by(step_size) {
            if i >= max_window_idx {
                println!("Reached end of safe window range at window {}", i);
                break;
            }

            spinner.set_message(format!("Listening for {} tone... (window {})", signal_name, i));
            spinner.tick();

            if let Some(freq) = self.detect_frequency(window, i) {
                let freq_match = (freq - target_frequency).abs() < FREQUENCY_TOLERANCE;
                
                println!("Window {}: Detected frequency = {:.1} Hz, Match = {}", i, freq, freq_match);
                
                if freq_match {
                    let start_pos = i * step_size;
                    spinner.finish_with_message(format!(
                        "{} signal detected at sample {}! (Frequency: {:.1} Hz)",
                        signal_name, start_pos, freq
                    ));
                    return Some(start_pos);
                }
            } else {
                println!("Window {}: No frequency detected", i);
            }

            if i % 10 == 0 {
                println!("Processed {} windows", i);
            }
        }

        spinner.finish_with_message(format!("No {} signal detected in current buffer", signal_name));
        None
    }

    fn decode_audio(&mut self, samples: &[f32]) -> Vec<u8> {
        let samples_per_bit = (self.sample_rate as f64 * (BIT_DURATION_MS as f64 / 1000.0)) as usize;
        self.stats.total_samples = samples.len();
        
        println!("\n{}", style("Decoding Parameters:").cyan());
        println!("├─ Sample rate: {} Hz", self.sample_rate);
        println!("├─ Bit duration: {} ms", BIT_DURATION_MS);
        println!("├─ Samples per bit: {}", samples_per_bit);
        println!("├─ Zero frequency: {:.1} Hz", self.frequencies.zero);
        println!("└─ One frequency: {:.1} Hz", self.frequencies.one);
        
        let mut decoded = Vec::new();
        let mut state = DecodeState::WaitingForSync;
        let mut sync_count = 0;
        let mut preamble_bits = 0;
        let mut expected_preamble = true;  // true for 1, false for 0
        
        println!("\n{}", style("Starting decode process:").cyan());
        
        for (i, chunk) in samples.chunks(samples_per_bit).enumerate() {
            if let Some(freq) = self.detect_frequency(chunk, i) {
                // println!("Chunk {}: {:.1} Hz", i, freq);
                
                match state {
                    DecodeState::WaitingForSync => {
                        if (freq - self.frequencies.start).abs() < FREQUENCY_TOLERANCE {
                            sync_count += 1;
                            println!("Sync marker {} detected", sync_count);
                            if sync_count >= 4 {
                                println!("{}", style("Sync sequence complete").green());
                                state = DecodeState::WaitingForPreamble;
                            }
                        } else {
                            if sync_count > 0 {
                                println!("Reset sync count (got {:.1} Hz)", freq);
                            }
                            sync_count = 0;
                        }
                    },
                    DecodeState::WaitingForPreamble => {
                        let is_one = (freq - self.frequencies.one).abs() < FREQUENCY_TOLERANCE;
                        let is_zero = (freq - self.frequencies.zero).abs() < FREQUENCY_TOLERANCE;
                        
                        if (is_one && expected_preamble) || (is_zero && !expected_preamble) {
                            preamble_bits += 1;
                            println!("Valid preamble bit {} of 16", preamble_bits);
                            expected_preamble = !expected_preamble;
                            if preamble_bits >= 16 {
                                println!("{}", style("Preamble detected").green());
                                state = DecodeState::WaitingForByteStart;
                            }
                        } else {
                            println!("Invalid preamble bit");
                            preamble_bits = 0;
                            expected_preamble = true;
                        }
                    },
                    DecodeState::WaitingForByteStart => {
                        if (freq - self.frequencies.one).abs() < FREQUENCY_TOLERANCE {
                            state = DecodeState::ReadingBits { 
                                current_byte: 0, 
                                bits_read: 0 
                            };
                            println!("Starting new byte");
                        }
                    },
                    DecodeState::ReadingBits { mut current_byte, bits_read } => {
                        let bit = if (freq - self.frequencies.one).abs() < FREQUENCY_TOLERANCE { 1 } else { 0 };
                        current_byte = (current_byte << 1) | bit;
                        
                        // println!("Read bit {} ({}/8)", bit, bits_read + 1);
                        
                        if bits_read == 7 {
                            state = DecodeState::WaitingForByteEnd;
                            decoded.push(current_byte);
                            if current_byte.is_ascii() {
                                println!("Completed byte: 0x{:02x} (ASCII: {})", 
                                    current_byte,
                                    if current_byte.is_ascii_control() { '.' } 
                                    else { current_byte as char });
                            } else {
                                println!("Completed byte: 0x{:02x}", current_byte);
                            }
                        } else {
                            state = DecodeState::ReadingBits { 
                                current_byte,
                                bits_read: bits_read + 1 
                            };
                        }
                    },
                    DecodeState::WaitingForByteEnd => {
                        if (freq - self.frequencies.zero).abs() < FREQUENCY_TOLERANCE {
                            state = DecodeState::WaitingForByteStart;
                        } else {
                            println!("Warning: Missing byte end marker");
                            // Continue anyway
                            state = DecodeState::WaitingForByteStart;
                        }
                    }
                }
                
                // Check for end marker
                if (freq - self.frequencies.end).abs() < FREQUENCY_TOLERANCE {
                    println!("{}", style("End marker detected").green());
                    break;
                }
            }
        }
        
        println!("\n{}", style("Decode Summary:").cyan());
        println!("├─ Decoded {} bytes", decoded.len());
        if !decoded.is_empty() {
            println!("└─ First byte: 0x{:02x}", decoded[0]);
        }
        
        decoded
    }
}


struct FileReceiver {
    decoder: AudioDecoder,
    output_dir: PathBuf,
}

impl FileReceiver {

    fn new(output_dir: PathBuf) -> Self {
        // Create output directory if it doesn't exist
        std::fs::create_dir_all(&output_dir).expect("Failed to create output directory");
        
        Self {
            decoder: AudioDecoder::new(SAMPLE_RATE),
            output_dir,
        }
    }

    fn debug_decoded_data(&self, data: &[u8]) {
        println!("\n{}", style("Decoded Data Analysis:").cyan().bold());
        println!("├─ Total bytes: {}", data.len());
        
        println!("├─ Pipe character (|) positions:");
        for (i, &byte) in data.iter().enumerate() {
            if byte == b'|' {
                println!("│  ├─ Position {}", i);
            }
        }
        
        println!("└─ First 100 bytes:");
        for chunk in data.iter().take(100).collect::<Vec<_>>().chunks(16) {
            print!("   ");
            for byte in chunk {
                print!("{:02x} ", byte);
            }
            
            for _ in chunk.len()..16 {
                print!("   ");
            }
            
            print!("  ");
            for &byte in chunk {
                if byte.is_ascii_graphic() || *byte == b' ' {
                    print!("{}", *byte as char);
                } else {
                    print!(".");
                }
            }
            println!();
        }
    }
    
    fn parse_header(&self, header_data: &[u8]) -> Result<(String, u64, String)> {
        let header_str = std::str::from_utf8(header_data)
            .context("Failed to parse header as UTF-8")?;
        
        println!("Raw header: {}", header_str);
        
        let parts: Vec<&str> = header_str.split('|').collect();
        if parts.len() < 3 {
            anyhow::bail!("Invalid header format: expected 3 parts, got {}", parts.len());
        }
        
        let filename = parts[0].to_string();
        let size = parts[1].parse().context("Failed to parse file size")?;
        let checksum = parts[2].to_string();
        
        Ok((filename, size, checksum))
    }
    
    fn verify_checksum(&self, data: &[u8], expected: &str) -> bool {
        let mut hasher = Sha256::new();
        hasher.update(data);
        let actual = hex::encode(hasher.finalize());
        
        println!("Checksum verification:");
        println!("├─ Expected: {}", style(expected).yellow());
        println!("└─ Actual  : {}", style(&actual).green());
        
        actual == expected
    }

    fn start_receiving(&mut self) -> Result<()> {
        println!("{}", style("Audio File Receiver").cyan().bold());
        
        let host = cpal::default_host();
        
        let device = host.input_devices()?
            .find(|device| {
                let name = device.name().ok().unwrap_or_default();
                name.to_lowercase().contains("blackhole")
            })
            .context("Could not find BlackHole audio device")?;
    
        println!("Using audio input device: {}", style(device.name()?).green());
    
        let config = device.default_input_config()?;
        println!("Device config: {:?}", config);
    
        let channels = config.channels() as usize;
        println!("Number of channels: {}", channels);
    
        let sample_counter = Arc::new(AtomicUsize::new(0));
        let sample_counter_clone = sample_counter.clone();
    
        let _running = Arc::new(AtomicBool::new(true));
        let rb = HeapRb::<f32>::new(SAMPLE_RATE as usize * 60 * 2 * channels);
        let (mut producer, mut consumer) = rb.split();
    
        let stream = device.build_input_stream(
            &config.into(),
            move |data: &[f32], _: &_| {
                if sample_counter_clone.load(std::sync::atomic::Ordering::Relaxed) % (SAMPLE_RATE as usize * 5) == 0 {
                    let max_amp = data.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
                    println!("Max amplitude in current buffer: {:.6}", max_amp);
                }
    
                let mono_samples: Vec<f32> = data.iter()
                    .step_by(channels)
                    .copied()
                    .collect();
            
                for &sample in &mono_samples {
                    if producer.push(sample).is_err() {
                        break;
                    }
                }
    
                sample_counter_clone.fetch_add(data.len(), std::sync::atomic::Ordering::Relaxed);
            },
            move |err| eprintln!("Error in audio stream: {}", err),
            None,
        )?;
    
        stream.play()?;
        println!("\n{}", style("Ready to receive transmission").yellow());
        println!("Listening for sync tone... (Press Ctrl+C to exit)");
    
        let mut all_samples = Vec::new();
        let mut last_debug_time = Instant::now();
        
        loop {
            std::thread::sleep(std::time::Duration::from_millis(100));
            
            let mut new_samples = Vec::new();
            while let Some(sample) = consumer.pop() {
                new_samples.push(sample);
            }
    
            if !new_samples.is_empty() && last_debug_time.elapsed() >= std::time::Duration::from_secs(1) {
                let max_amp = new_samples.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
                let rms = (new_samples.iter().map(|&x| x * x).sum::<f32>() / new_samples.len() as f32).sqrt();
                println!("Buffer stats: {} samples, max amp: {:.6}, RMS: {:.6}", 
                        new_samples.len(), max_amp, rms);
                last_debug_time = Instant::now();
            }
    
            all_samples.extend(new_samples);
    
            if all_samples.len() >= SAMPLE_RATE as usize * 3 {
                let rms = (all_samples.iter().map(|&x| x * x).sum::<f32>() / all_samples.len() as f32).sqrt();
                println!("Full buffer stats before sync detection:");
                println!("├─ Samples: {}", all_samples.len());
                println!("└─ RMS amplitude: {:.6}", rms);
    
                if let Some(sync_pos) = self.decoder.wait_for_sync(&all_samples) {
                    println!("\n{}", style("Sync detected! Waiting for preamble...").green());
                    
                    // Keep collecting samples for a short while to ensure we have the preamble
                    let mut samples_after_sync = all_samples[sync_pos..].to_vec();
                    let start_time = Instant::now();
                    
                    while start_time.elapsed() < Duration::from_secs(2) {
                        while let Some(sample) = consumer.pop() {
                            samples_after_sync.push(sample);
                        }
                        thread::sleep(Duration::from_millis(100));
                    }
    
                    if let Some(preamble_end) = self.decoder.wait_for_preamble(&samples_after_sync, 0) {
                        all_samples = samples_after_sync[preamble_end..].to_vec();
                        println!("\n{}", style("Preamble detected! Receiving file...").green());
                        
                        let start_time = Instant::now();
                        while start_time.elapsed() < Duration::from_secs(1000) {
                            while let Some(sample) = consumer.pop() {
                                all_samples.push(sample);
                            }
                            std::thread::sleep(Duration::from_millis(30));
                        }
                    
                        println!("\nRecording complete:");
                        println!("Captured {} samples ({:.2} seconds)",
                                all_samples.len(),
                                all_samples.len() as f32 / SAMPLE_RATE as f32);
    
                        let decoded_data = self.decoder.decode_audio(&all_samples);
                        
                        if decoded_data.is_empty() {
                            anyhow::bail!("No data decoded from audio signal");
                        }
    
                        println!("\n{}", style("Looking for header in decoded data...").yellow());
                        self.debug_decoded_data(&decoded_data);
    
                        if let Some(first_pipe) = decoded_data.iter().position(|&b| b == b'|') {
                            if let Some(second_pipe_offset) = decoded_data[first_pipe + 1..].iter().position(|&b| b == b'|') {
                                let second_pipe = first_pipe + 1 + second_pipe_offset;
                                println!("\nFound header structure:");
                                println!("├─ First pipe at: {}", first_pipe);
                                println!("└─ Second pipe at: {}", second_pipe);
                                
                                // Search for |END| marker in the entire decoded data
                                if let Some(header_end) = decoded_data.windows(5)
                                    .enumerate()
                                    .find(|(_, window)| window == b"|END|")
                                    .map(|(pos, _)| pos) 
                                {
                                    println!("Found |END| marker at position {}", header_end);
                                    
                                    // Verify the header structure makes sense
                                    if header_end > second_pipe {
                                        let header_slice = &decoded_data[..header_end];
                                        println!("Found header of {} bytes", header_slice.len());
                                        println!("Header content: {:?}", std::str::from_utf8(header_slice).unwrap_or("Invalid UTF-8"));
                                        
                                        if let Ok((filename, size, checksum)) = self.parse_header(header_slice) {
                                            println!("\nHeader details:");
                                            println!("├─ Filename: {}", style(&filename).green());
                                            println!("├─ Size: {} bytes", style(size).green());
                                            println!("└─ Checksum: {}", style(&checksum).green());
                                            
                                            // Skip the header and the |END| marker
                                            let file_data = &decoded_data[header_end + 5..];
                                            if file_data.len() < size as usize {
                                                println!("Warning: Not enough data after header ({} < {} bytes)", 
                                                        file_data.len(), size);
                                                return Err(anyhow::anyhow!("Incomplete file data"));
                                            }
                                            
                                            let file_data = &file_data[..size as usize];
                                            
                                            if self.verify_checksum(file_data, &checksum) {
                                                let output_path = self.output_dir.join(&filename);
                                                
                                                // Create parent directories if they don't exist
                                                if let Some(parent) = output_path.parent() {
                                                    std::fs::create_dir_all(parent)
                                                        .context("Failed to create parent directories")?;
                                                }
                                                
                                                // Write the file
                                                std::fs::write(&output_path, file_data)
                                                    .with_context(|| format!("Failed to write file to {}", output_path.display()))?;
                                                
                                                println!("\n{}", style("File successfully saved!").green().bold());
                                                println!("Location: {}", style(output_path.display()).green());
                                                self.decoder.stats.print_summary();
                                                return Ok(());
                                            } else {
                                                println!("{}", style("Checksum verification failed").red());
                                            }
                                        } else {
                                            return Err(anyhow::anyhow!("Failed to parse header"));
                                        }
                                    } else {
                                        return Err(anyhow::anyhow!("Invalid header structure: END marker before second pipe"));
                                    }
                                } else {
                                    return Err(anyhow::anyhow!("Could not find |END| marker"));
                                }
                            } else {
                                return Err(anyhow::anyhow!("Could not find second pipe character"));
                            }
                        } else {
                            return Err(anyhow::anyhow!("Could not find first pipe character"));
                        }
                    }
                }
                
                // If sync not found, keep only the last 2 seconds of samples
                let keep_samples = SAMPLE_RATE as usize * 2;
                if all_samples.len() > keep_samples {
                    all_samples = all_samples[all_samples.len() - keep_samples..].to_vec();
                }
            }
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut receiver = FileReceiver::new(args.output_dir);
    
    println!("{}", style("Starting file receiver...").cyan().bold());
    receiver.start_receiving()?;
    
    Ok(())
}

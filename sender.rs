use std::path::PathBuf;
use std::time::Duration;
use std::thread;
use anyhow::{Result, Context};
use clap::Parser;
use rodio::{OutputStream, Sink, Source};
use sha2::{Sha256, Digest};
use indicatif::{ProgressBar, ProgressStyle};
use console::style;
use reed_solomon::Encoder;

const SAMPLE_RATE: u32 = 48000;
const BIT_DURATION_MS: u64 = 2; 
const AMPLITUDE: f32 = 0.8;
const SYNC_DURATION_MS: u64 = 1000;
const SYNC_FREQUENCY: f32 = 240.0;
const PREAMBLE_DURATION_MS: u64 = 500;
const PREAMBLE_FREQUENCY: f32 = 600.0;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input file to transmit
    #[arg(short, long)]
    input: PathBuf,
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

struct TransmissionStats {
    total_bytes: usize,
    start_time: std::time::Instant,
}

impl TransmissionStats {
    fn new(total_bytes: usize) -> Self {
        Self {
            total_bytes,
            start_time: std::time::Instant::now(),
        }
    }

    fn print_summary(&self) {
        let duration = self.start_time.elapsed();
        let rate = (self.total_bytes as f64) / duration.as_secs_f64();
        
        println!("\n{}", style("Transmission Summary:").cyan().bold());
        println!("├─ Total bytes: {}", style(self.total_bytes).green());
        println!("├─ Duration: {:.2}s", style(duration.as_secs_f64()).green());
        println!("└─ Transfer rate: {:.2} bytes/sec", style(rate).green());
    }
}

struct SineWaveSource {
    frequency: f32,
    sample_rate: u32,
    amplitude: f32,
    num_samples: usize,
    current_sample: usize,
}

impl SineWaveSource {
    fn new(frequency: f32, duration_ms: u64, sample_rate: u32, amplitude: f32) -> Self {
        let num_samples = (sample_rate as f64 * (duration_ms as f64 / 1000.0)) as usize;
        Self {
            frequency,
            sample_rate,
            amplitude,
            num_samples,
            current_sample: 0,
        }
    }
}

impl Iterator for SineWaveSource {
    type Item = f32;

    fn next(&mut self) -> Option<f32> {
        if self.current_sample >= self.num_samples {
            return None;
        }

        let t = self.current_sample as f32 / self.sample_rate as f32;
        let sample = (2.0 * std::f32::consts::PI * self.frequency * t).sin() * self.amplitude;
        self.current_sample += 1;
        Some(sample)
    }
}

impl Source for SineWaveSource {
    fn current_frame_len(&self) -> Option<usize> {
        Some(self.num_samples - self.current_sample)
    }

    fn channels(&self) -> u16 {
        1
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn total_duration(&self) -> Option<std::time::Duration> {
        Some(std::time::Duration::from_millis(
            (self.num_samples as f64 / self.sample_rate as f64 * 1000.0) as u64,
        ))
    }
}

struct AudioEncoder {
    sample_rate: u32,
    frequencies: FrequencyMap,
    progress_bar: ProgressBar,
    stats: TransmissionStats,
}

impl AudioEncoder {
    fn new(sample_rate: u32, total_bytes: usize) -> Self {
        let progress_bar = ProgressBar::new(total_bytes as u64);
        progress_bar.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
                .expect("Invalid progress bar template")
                .progress_chars("#>-")
        );

        Self {
            sample_rate,
            frequencies: FrequencyMap::default(),
            progress_bar,
            stats: TransmissionStats::new(total_bytes),
        }
    }

    fn send_sync_sequence(&self, sink: &Sink) {
        println!("{}", style("\nSending sync sequence...").yellow());

        // Send silence to separate sync from data
        sink.append(SineWaveSource::new(0.0, 400, self.sample_rate, 0.0));
        
        // Send sync tone (longer duration)
        sink.append(SineWaveSource::new(
            SYNC_FREQUENCY,
            SYNC_DURATION_MS,
            self.sample_rate,
            AMPLITUDE
        ));
        
        // Send silence to separate sync from data
        sink.append(SineWaveSource::new(0.0, 200, self.sample_rate, 0.0));
        
        // Wait for sync to complete
        thread::sleep(Duration::from_millis(SYNC_DURATION_MS + 400));
    }

    fn send_preamble(&self, sink: &Sink) {
        println!("{}", style("\nSending preamble...").yellow());
        
        // Send preamble tone
        sink.append(SineWaveSource::new(
            PREAMBLE_FREQUENCY,
            PREAMBLE_DURATION_MS,
            self.sample_rate,
            AMPLITUDE
        ));
        
        // Send silence to separate preamble from data
        sink.append(SineWaveSource::new(0.0, 400, self.sample_rate, 0.0));
        
        // Wait for preamble to complete
        thread::sleep(Duration::from_millis(PREAMBLE_DURATION_MS + 800));
    }

    fn encode_bytes(&mut self, data: &[u8], sink: &Sink) {
        // Initial silence
        sink.append(SineWaveSource::new(
            0.0,
            BIT_DURATION_MS * 4,
            self.sample_rate,
            0.0
        ));
    
        // Synchronization sequence
        for _ in 0..4 {  // Increased from 3 to 4
            // Start marker
            sink.append(SineWaveSource::new(
                self.frequencies.start,
                BIT_DURATION_MS * 4,
                self.sample_rate,
                AMPLITUDE
            ));
            
            // Silence gap
            sink.append(SineWaveSource::new(
                0.0,
                BIT_DURATION_MS * 2,
                self.sample_rate,
                0.0
            ));
        }
    
        // Preamble - fixed pattern 0x55 (alternating 1s and 0s)
        for _ in 0..8 {
            sink.append(SineWaveSource::new(
                self.frequencies.one,
                BIT_DURATION_MS,
                self.sample_rate,
                AMPLITUDE
            ));
            
            sink.append(SineWaveSource::new(
                self.frequencies.zero,
                BIT_DURATION_MS,
                self.sample_rate,
                AMPLITUDE
            ));
        }

        // Add a clear separator after the header
        let header_end_marker = b"|END|";
        let mut transmission_data = Vec::new();
        
        // Add header
        transmission_data.extend_from_slice(data);
        transmission_data.extend_from_slice(header_end_marker);
        
        // Add extra silence between header and data
        sink.append(SineWaveSource::new(
            0.0,
            BIT_DURATION_MS * 8,
            self.sample_rate,
            0.0
        ));
    
        // Data with byte boundaries
        for (i, &byte) in data.iter().enumerate() {
            // Byte start marker
            sink.append(SineWaveSource::new(
                self.frequencies.one,
                BIT_DURATION_MS,
                self.sample_rate,
                AMPLITUDE
            ));
    
            // Data bits
            for bit_pos in (0..8).rev() {
                let bit = (byte >> bit_pos) & 1;
                let freq = if bit == 1 {
                    self.frequencies.one
                } else {
                    self.frequencies.zero
                };
                
                sink.append(SineWaveSource::new(
                    freq,
                    BIT_DURATION_MS,
                    self.sample_rate,
                    AMPLITUDE
                ));
            }
    
            // Byte end marker
            sink.append(SineWaveSource::new(
                self.frequencies.zero,
                BIT_DURATION_MS,
                self.sample_rate,
                AMPLITUDE
            ));
            
            self.progress_bar.set_position((i + 1) as u64);
        }
    
        // End sequence
        sink.append(SineWaveSource::new(
            0.0,
            BIT_DURATION_MS * 16,
            self.sample_rate,
            0.0
        ));
        
        for _ in 0..4 {
            sink.append(SineWaveSource::new(
                self.frequencies.end,
                BIT_DURATION_MS * 16,
                self.sample_rate,
                AMPLITUDE * 1.2
            ));
            
            sink.append(SineWaveSource::new(
                0.0,
                BIT_DURATION_MS * 2,
                self.sample_rate,
                0.0
            ));
        }
    }
}

struct FileTransfer {
    encoder: AudioEncoder,
}

impl FileTransfer {
    fn new(total_bytes: usize) -> Self {
        Self {
            encoder: AudioEncoder::new(SAMPLE_RATE, total_bytes),
        }
    }

    fn create_header(&self, filename: &str, file_size: u64, checksum: &str) -> String {
        format!("{}|{}|{}", filename, file_size, checksum)
    }

    fn calculate_checksum(&self, data: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(data);
        hex::encode(hasher.finalize())
    }

    fn transmit_file(&mut self, input_path: &PathBuf) -> Result<()> {
        println!("{}", style("\nInitializing file transfer...").cyan().bold());
        
        // Read file
        let file_data = std::fs::read(input_path)
            .context("Failed to read input file")?;
        
        // Create header
        let filename = input_path.file_name()
            .context("Invalid filename")?
            .to_string_lossy();
        let checksum = self.calculate_checksum(&file_data);
        let header = self.create_header(&filename, file_data.len() as u64, &checksum);
        
        println!("File: {}", style(&filename).green());
        println!("Size: {} bytes", style(file_data.len()).green());
        println!("Checksum: {}", style(&checksum).green());
        
        // Initialize audio output
        let (_stream, stream_handle) = OutputStream::try_default()
            .context("Failed to initialize audio output")?;
        let sink = Sink::try_new(&stream_handle)
            .context("Failed to create audio sink")?;
    
        println!("\n{}", style("Waiting for receiver...").yellow());
        println!("Press Enter when ready to start transmission");
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
    
        // Send sync sequence
        self.encoder.send_sync_sequence(&sink);
        
        // Send preamble
        self.encoder.send_preamble(&sink);
        
        println!("{}", style("\nStarting transmission...").green());
        
        // Prepare the complete transmission data
        let mut transmission_data = Vec::new();
        
        // Add header
        transmission_data.extend_from_slice(header.as_bytes());
        
        // Add |END| marker after header
        transmission_data.extend_from_slice(b"|END|");
        
        // Add file data
        transmission_data.extend_from_slice(&file_data);
    
        // Apply Reed-Solomon encoding
        let rs = Encoder::new(32);
        let encoded_data: Vec<u8> = transmission_data
            .chunks(223)
            .flat_map(|chunk| {
                let mut padded_chunk = chunk.to_vec();
                padded_chunk.resize(223, 0); // Pad with zeros if necessary
                rs.encode(&padded_chunk).data().to_vec()
            })
            .collect();
    
        // Debug print the first part of the transmission data
        println!("\nTransmission data preview:");
        for (i, chunk) in transmission_data[..100.min(transmission_data.len())].chunks(16).enumerate() {
            print!("{:04x}: ", i * 16);
            for byte in chunk {
                print!("{:02x} ", byte);
            }
            print!("  ");
            for &byte in chunk {
                print!("{}", if byte.is_ascii_graphic() || byte == b' ' {
                    byte as char
                } else {
                    '.'
                });
            }
            println!();
        }
    
        self.encoder.encode_bytes(&encoded_data, &sink);
        
        sink.sleep_until_end();
        self.encoder.stats.print_summary();
        
        println!("\n{}", style("Transmission complete!").green().bold());
    
        Ok(())
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let total_bytes = std::fs::metadata(&args.input)?.len() as usize;
    let mut transfer = FileTransfer::new(total_bytes);
    
    println!("{}", style("Audio File Transfer").cyan().bold());
    transfer.transmit_file(&args.input)?;
    
    Ok(())
}

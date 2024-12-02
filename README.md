# Rusty-Telephone 🎧

Data exfiltration tool that transmits files through audio loopback devices using frequency-shift keying (FSK) modulation.

## Overview

Rusty-Telephone encodes files into audio signals and transmits them between systems using audio loopback. It uses:
- FSK modulation with multiple frequencies for data encoding
- Reed-Solomon error correction
- SHA-256 checksums for data integrity
- Sync sequences and preambles for reliable transmission
- Digital signal processing for audio analysis

## Prerequisites

- Rust toolchain (latest stable)
- Audio loopback driver:
  - macOS: [BlackHole](https://github.com/ExistentialAudio/BlackHole)
  - Windows: [Virtual Audio Cable](https://vb-audio.com/Cable/)
  - Linux: [PulseAudio null sink](https://www.freedesktop.org/wiki/Software/PulseAudio/)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/referefref/rusty-telephone
cd rusty-telephone
```

2. Build sender and receiver:

**macOS/Linux:**
```bash
cargo build --release -p sender
cargo build --release -p receiver
```

**Windows:**
```bash
cargo build --release -p sender
cargo build --release -p receiver
```

Binaries will be in `target/release/`.

## Usage

1. Set up audio loopback:
   - **macOS**: Install BlackHole or similar and select it as output device
   - **Windows**: Install Virtual Audio Cable or similar and configure it
   - **Linux**: Create null sink: `pactl load-module module-null-sink sink_name=loopback`

2. Start receiver:
```bash
./receiver --output-dir /path/to/output
```

3. Start sender:
```bash
./sender --input /path/to/file
```

4. Press Enter in sender window to begin transmission

## Technical Details

- Sample rate: 48kHz
- Bit duration: 2ms
- FSK frequencies:
  - Sync: 240Hz
  - Preamble: 600Hz
  - Data '0': 2000Hz
  - Data '1': 4000Hz
  - Start marker: 1000Hz
  - End marker: 6000Hz

### Dependencies

```toml
# sender/Cargo.toml
[dependencies]
rodio = "0.17"
bytes = "1.4"
sha2 = "0.10"
hex = "0.4"
anyhow = "1.0"
clap = { version = "4.3", features = ["derive"] }
indicatif = "0.17"
console = "0.15"
reed-solomon = "0.2.1"

# receiver/Cargo.toml
[dependencies]
cpal = "0.15"
rustfft = "6.1"
bytes = "1.4"
sha2 = "0.10"
hex = "0.4"
anyhow = "1.0"
clap = { version = "4.3", features = ["derive"] }
ringbuf = "0.3"
indicatif = "0.17"
console = "0.15"
reed-solomon = "0.2.1"
```

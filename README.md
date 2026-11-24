<h1 align="center">Wasabi</h1>
<p align="center"><img src="/assets/logo.svg" width="128"/></p>
<p align="center">Wasabi is a modern and fast real-time MIDI player written in Rust.</p>
<p align="center">
<img alt="GitHub License" src="https://img.shields.io/github/license/BlackMIDIDevs/wasabi">
<img alt="GitHub Release" src="https://img.shields.io/github/v/release/BlackMIDIDevs/wasabi">
<img alt="GitHub Downloads (all assets, all releases)" src="https://img.shields.io/github/downloads/BlackMIDIDevs/wasabi/total">
</p>

## Features

- Extremely fast and optimized rendering using Vulkan
- Easy to use and configurable
- Integrated MIDI synthesizer (XSynth), alongside with KDMAPI and MIDI device support
- Partial support for Zenith color palettes

## Installation

Wasabi is a portable application and does not require an installation.
Your system must support Vulkan.

### Option A *(recommended)*

You can download and run a pre-built binary of Wasabi from the [releases page.](https://github.com/BlackMIDIDevs/wasabi/releases)

### Option B *(advanced)*

You can build Wasabi yourself by following these steps:

- Clone the repository using `git clone https://github.com/BlackMIDIDevs/wasabi.git` (or [download as a ZIP from GitHub](https://github.com/BlackMIDIDevs/wasabi/archive/refs/heads/master.zip))
- Required tools:
    - [Rust toolchain](https://rustup.rs/)
    - [Vulkan SDK](https://vulkan.lunarg.com/)
    - [CMake **3.X**](https://cmake.org/)
    - [Ninja](https://ninja-build.org/)
    - (C++ build tools for-)[Visual Studio 17+](https://visualstudio.microsoft.com/) (Windows only)
- Inside the project directory run the following command to build Wasabi: `cargo build --release`
    - Optionally you can add `RUSTFLAGS="-C target-cpu=native"` to your environment before compiling to optimize XSynth for your specific CPU
- After the compilation is finished, you will find the binary under `./target/release`

## Usage

- Before you can play a MIDI, you need to add soundfonts to the synthesizer by going to `Menu -> Settings -> SoundFonts`
- To open a MIDI, click the folder icon on the top left, or press `Ctrl+O` on your keyboard
- To find out about other keyboard shortcuts, head to `Menu -> Shortcuts`

## Screenshot

<p align="center"><img src="/assets/screenshot.png"/></p>

## License
Wasabi is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html#license-text).

---

## Note from the Author

Hi! I'm just an ordinary person living in a small city in South Korea.  
This project is **based on Wasabi**, and I intentionally left the original Wasabi branding and preview assets untouched out of respect for the original developers. All engine tuning, lock-free `kdmapiv3` integration, and optimization work were crafted via CLI with AI assistance.

I deeply love and respect everyone in the Black MIDI community—original Wasabi developers, synth engineers, MIDI creators, and players alike—and I hold no ill will toward anyone. This was built out of pure passion for exploring real-time playback limits :)

---

안녕하세요! 대한민국에 살고 있는 한 소도시 사람입니다.  
이 프로젝트는 **Wasabi를 기반**으로 제작되었으며, 원작 개발자분들에 대한 존중의 의미로 기존 Wasabi 로고와 미리보기 리소스는 수정하지 않고 그대로 남겨두었습니다. 엔진 튜닝과 락프리 `kdmapiv3` 직결, 최적화 작업은 AI의 도움을 받아 CLI 환경에서 작업했습니다.

원본 Wasabi 개발자분들을 비롯한 검은악보(Black MIDI) 관련자분들 모두를 사랑하고 존경하며, 어떠한 악감정도 없습니다. 순수한 애정으로 한계 테스트를 위해 만든 프로젝트입니다 :)

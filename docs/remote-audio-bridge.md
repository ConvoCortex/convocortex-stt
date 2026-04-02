# Remote audio bridge workflow

This document describes the currently intended way to use `convocortex-stt`:
phone as the normal microphone, headphones as the normal local output, and
automatic fallback to phone playback when the headphones disconnect.

STT itself stays local on the desktop. The phone is the remote audio endpoint.

The exact network product is not important. What matters is that the remote
device can reach the desktop over a routed LAN, overlay VPN, or similar remote
access path.

## Components

Typical setup:
- Windows desktop running `convocortex-stt`
- Mumble-compatible client/server path
- virtual audio devices such as Voicemeeter Banana
- remote client such as a phone running a Mumble-compatible app

## Goal

Use the desktop as the STT host while letting the phone provide:
- the normal microphone path into the desktop
- the fallback speaker path from the desktop

This is not mainly about manually switching STT devices.
It is about making the audio stack itself do the right thing automatically so
desk use, room-to-room use, and remote use all feel like the same setup.

## Conceptual routing

Remote microphone path:
- remote mic -> remote Mumble client -> network -> desktop Mumble client/server
- desktop Mumble output -> `Voicemeeter Aux Input`
- `Voicemeeter Aux Input` -> `B1`
- STT input device = `B1`

Desktop playback fallback path:
- Windows output -> `Voicemeeter Input`
- `Voicemeeter Input` -> `B2`
- desktop Mumble input = `B2`
- network -> remote Mumble client playback

Normal local playback path:
- Windows output -> local headphones

The important behavior is:
- if headphones are on, local desktop audio should stay on the headphones
- if headphones are turned off or disconnected, Windows should fall back to `Voicemeeter Input`
- once Windows falls back to `Voicemeeter Input`, desktop audio should start reaching the phone automatically

## Mumble and virtual audio settings

Desktop Mumble:
- audio input = `Voicemeeter Out B2`
- audio output = `Voicemeeter Aux Input`
- transmission:
  use voice-activated transmission for normal remote-mic use, but lower the voice-activity / speech-detection thresholds aggressively so short STT feedback sounds still get transmitted
- practical starting point:
  roughly `15%` for sound activity and `25%` for speech activity, then tune by ear
- if continuous transmission works but voice-activated mode drops feedback sounds, the issue is Mumble activation thresholds rather than the STT feedback path itself
- connect to the local desktop Mumble server automatically on startup

Desktop Mumble server:
- run automatically when the PC starts
- listen on the local machine / LAN / overlay network path you actually use
- stay stable enough that the phone can reconnect without manual server babysitting

Phone Mumble client:
- connect to the desktop Mumble server
- reconnect when needed after network changes or phone-call interruptions
- after a call ends, the Mumble audio path should resume so the phone keeps acting as the PC microphone/speaker endpoint

Voicemeeter Banana:
- `Voicemeeter Input` -> `B2`
- `Voicemeeter Aux Input` -> `B1`

## Windows device defaults

For the current intended workflow:
- Windows default input = `Voicemeeter Out B1`
- Windows primary output = your normal headphones
- Windows fallback output = `Voicemeeter Input`

The desired behavior is that Windows itself handles the output transition:
- headphones connected / active -> playback stays on headphones
- headphones disconnected / powered off -> playback falls back to `Voicemeeter Input`

That way a single physical action, turning the headphones off, moves desktop
audio to the phone route without needing STT to switch output devices.

## STT device expectations

For this setup, STT device cycling is not the primary control surface.

Recommended STT expectations:
- STT input should normally stay on `Voicemeeter Out B1`
- STT feedback/output should normally stay on the desktop headphones or on the Windows-selected default path
- STT input/output cycle commands are optional fallback tools, not the normal daily path

If you never intentionally switch away from the phone-mic path, disabling the
input-cycle voice command words is reasonable.

## Startup automation

The intended startup behavior is:
- desktop Mumble server starts automatically with Windows
- desktop Mumble client starts automatically with Windows
- desktop Mumble client connects to the desktop Mumble server automatically
- Voicemeeter starts with the saved routing already in place
- Windows starts with the expected default / fallback audio devices already configured
- STT starts and uses the bridge path without needing manual device changes

If those pieces are correct, the whole stack behaves like one continuous audio
system instead of several apps that need to be reconfigured during the day.

## Notes

- Avoid disabling or re-enabling devices in Windows Device Manager while the
  stack is already running. That can leave Windows defaults and app device
  bindings in a bad state until restart.
- The point of this stack is low transition cost. The same phone should work as
  the microphone at the desk, elsewhere in the building, and remotely, without
  changing the core STT configuration.

# Remote audio bridge workflow

This document describes a practical way to use `convocortex-stt` with a remote
microphone and speaker path while still running STT locally on the desktop.

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

Use the desktop as the STT host while letting a remote client provide:
- microphone audio into the desktop
- speaker playback from the desktop

## Conceptual routing

Remote microphone path:
- remote mic -> remote Mumble client -> network -> desktop Mumble client/server
- desktop Mumble output -> `Voicemeeter Aux Input`
- `Voicemeeter Aux Input` -> `B1`
- STT input device = `B1`

Desktop playback path:
- Windows output -> `Voicemeeter Input`
- `Voicemeeter Input` -> `B2`
- desktop Mumble input = `B2`
- network -> remote Mumble client playback

## Mumble and virtual audio settings

Desktop Mumble:
- audio input = `Voicemeeter Out B2`
- audio output = `Voicemeeter Aux Input`
- transmission:
  use voice-activated transmission for normal remote-mic use, but lower the voice-activity / speech-detection thresholds aggressively so short STT feedback sounds still get transmitted
- practical starting point:
  roughly `15%` for sound activity and `25%` for speech activity, then tune by ear
- if continuous transmission works but voice-activated mode drops feedback sounds, the issue is Mumble activation thresholds rather than the STT feedback path itself

Voicemeeter Banana:
- `Voicemeeter Input` -> `B2`
- `Voicemeeter Aux Input` -> `B1`

## Windows device defaults

For remote-phone mode:
- Windows default output = `Voicemeeter Input`
- Windows default input = `Voicemeeter Out B1`

For local-PC mode:
- Windows default output = your normal speakers/headphones
- Windows default input = your normal desktop microphone

## Recommended STT input cycle

A useful input-device cycle list is:
- `Voicemeeter Out B1`
- your local desktop microphone
- optional headset microphone

That lets you compare or swap between:
- remote phone mic
- desk mic
- headset mic

## Recommended STT output cycle

A useful output-device cycle list is:
- `Voicemeeter Input`
- your normal headphones/speakers

That keeps the two practical playback targets available:
- remote-device route through Voicemeeter
- local listening on the desktop

## Notes

- Avoid disabling or re-enabling devices in Windows Device Manager while the
  stack is already running. That can leave Windows defaults and app device
  bindings in a bad state until restart.

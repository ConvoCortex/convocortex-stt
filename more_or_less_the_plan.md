What the STT binary is
A voice input runtime. Self-sufficient, works standalone, participates in NATS as a peer. Not a dump pipe, not a monolith. Owns the microphone and the transcription engine. Nothing else is permanent.

Local convenience features
Type at cursor. Clipboard replace. Clipboard accumulate. Append to file. Overwrite watched file. Trailing character per utterance. All default-on, all individually disableable via config. Reference implementation and fallback. Disabled when NATS consumers replace them. Not meant to be permanently relied on.

Voice command handling
No voice command engine. One exception: the emission gate. Configurable trigger words that gate whether transcription output is forwarded or discarded. Not a voice command engine — a string match on two words controlling a boolean. Inseparable from the pipeline. Everything else belongs to ConvoCortex.

NATS emit surface
Final results. Interim results. VAD state changes. Command events. System events — device changed, model changed, session start, session stop, error. Not audio metering. Not internal state transitions. Emit when an external consumer could reasonably act on it.

NATS control surface — permanent
Start, stop, push-to-talk. Device selection. Model and language. VAD sensitivity and thresholds. Wake word configuration. Emission gate trigger words. Status query. Shutdown. These are permanent because only STT can do them.

Local handler toggles
Config file only. Set before startup. Not runtime toggleable. Not on the control surface. One-time migration step when a NATS consumer replaces a local handler. Done in config, never touched again.

State persistence
Read state file once on startup. Never again during runtime. Every state change from any source writes to state file atomically as side effect. Restart restores state transparently. No watchdog, no hot reload, no feedback loops. File is crash recovery artifact not control mechanism.

Control entry points
Two only. Hotkeys — maximum four, instant human input, emission gate toggle, push to talk, maybe device cycle. NATS control subject — everything else when NATS is running. No CLI, no GUI, no TUI, no single instance socket, no signals.

Standalone mode without NATS
Not crippled. Transcription, output, hotkeys, audio feedback, emission gate all work. Missing only runtime config changes and external controllability. Hotkeys are enough for standalone use. If you need runtime control, run NATS.

Internal message schema
No internal bus. One message type constructed at origin, passed directly to handler functions. NATS publisher is one handler among others. Schema identical inside and outside the process. No impedance mismatch ever.

No GUI
Headless always. First line of the features section in the readme.

Full arc
Ship with local handlers enabled. Build NATS consumers progressively. Disable local handlers as consumers replace them. Arrive at pure pipe mode incrementally. STT binary never changes throughout. Only config changes.
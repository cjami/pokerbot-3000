"""Voice input and output adapters."""

from pokerbot_3000.voice.asr import (
    DEFAULT_PARAKEET_MODEL,
    POKERBOT_VOICE_MODEL_ENV,
    ParakeetConfig,
    ParakeetSpeechTranscriber,
)
from pokerbot_3000.voice.capture import (
    POKERBOT_VAD_MAX_PHRASE_MS_ENV,
    POKERBOT_VAD_MIN_PHRASE_MS_ENV,
    POKERBOT_VAD_THRESHOLD_ENV,
    POKERBOT_VOICE_DEVICE_ENV,
    MicrophoneConfig,
    SileroVoiceActivityDetector,
    SoundDeviceAudioInput,
    VadConfig,
    VoiceRuntimeError,
)
from pokerbot_3000.voice.coordinator import VoiceActionAdapters, VoiceActionCoordinator, VoiceInputStatus
from pokerbot_3000.voice.elevenlabs import (
    ELEVENLABS_API_KEY_ENV,
    ELEVENLABS_MODEL_ENV,
    ELEVENLABS_ORCHESTRATOR_SPEED_ENV,
    ELEVENLABS_ORCHESTRATOR_VOICE_ID_ENV,
    ElevenLabsClient,
    ElevenLabsClientError,
    ElevenLabsConfig,
    ElevenLabsConfigurationError,
)
from pokerbot_3000.voice.grammar import DeterministicVoiceCommandParser

__all__ = [
    "DEFAULT_PARAKEET_MODEL",
    "ELEVENLABS_API_KEY_ENV",
    "ELEVENLABS_MODEL_ENV",
    "ELEVENLABS_ORCHESTRATOR_SPEED_ENV",
    "ELEVENLABS_ORCHESTRATOR_VOICE_ID_ENV",
    "POKERBOT_VAD_MAX_PHRASE_MS_ENV",
    "POKERBOT_VAD_MIN_PHRASE_MS_ENV",
    "POKERBOT_VAD_THRESHOLD_ENV",
    "POKERBOT_VOICE_DEVICE_ENV",
    "POKERBOT_VOICE_MODEL_ENV",
    "DeterministicVoiceCommandParser",
    "ElevenLabsClient",
    "ElevenLabsClientError",
    "ElevenLabsConfig",
    "ElevenLabsConfigurationError",
    "MicrophoneConfig",
    "ParakeetConfig",
    "ParakeetSpeechTranscriber",
    "SileroVoiceActivityDetector",
    "SoundDeviceAudioInput",
    "VadConfig",
    "VoiceActionAdapters",
    "VoiceActionCoordinator",
    "VoiceInputStatus",
    "VoiceRuntimeError",
]

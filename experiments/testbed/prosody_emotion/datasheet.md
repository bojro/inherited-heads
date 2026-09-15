# V15 prosody strips — emotion axis

Source: RAVDESS (quinnlue/ravdess_emotional_speech_audio), 24 actors x 8
emotions x 2 statements x 2 repetitions, 16 kHz. 150 items, 6
segments each, built with seed 0.

**All six segments of a strip speak the identical sentence.** Segment identity is
carried only by emotional prosody, so
transcript-overlap attribution is meaningless by construction — score with a
emotion judge instead.

Conservative preprocessing, both of which make the test HARDER:
- every segment RMS-normalised to 0.05 (peak-guarded), so loudness cannot
  carry identity — decisive on the emotion axis, where angry is naturally louder
  than sad;
- silence trimmed at 35.0 dB below clip peak with a 50 ms margin, so the
  recorded boundaries bracket speech, not room tone.
- intensity pinned to normal (01), because neutral exists only at that intensity and mixing intensities would reintroduce the level cue.

Strip duration 10.8-18.2 s (median 14.0), under
the 30 s Whisper-family encoder window.

Columns: item_id, seg_idx, speaker_id, transcript, start_s, end_s, start_sample,
end_sample, axis, seg_label, seg_attr, emotion, intensity, statement,
repetition, gender, ravdess_file.

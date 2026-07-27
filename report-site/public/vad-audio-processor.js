class VadAudioProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.chunkSamples = Math.max(128, Math.round(sampleRate / 100));
    this.buffer = new Float32Array(this.chunkSamples);
    this.offset = 0;
  }

  process(inputs) {
    const channel = inputs[0]?.[0];
    if (!channel) {
      return true;
    }

    let sourceOffset = 0;
    while (sourceOffset < channel.length) {
      const count = Math.min(
        channel.length - sourceOffset,
        this.buffer.length - this.offset,
      );
      this.buffer.set(
        channel.subarray(sourceOffset, sourceOffset + count),
        this.offset,
      );
      sourceOffset += count;
      this.offset += count;

      if (this.offset === this.buffer.length) {
        this.port.postMessage(this.buffer, [this.buffer.buffer]);
        this.buffer = new Float32Array(this.chunkSamples);
        this.offset = 0;
      }
    }
    return true;
  }
}

registerProcessor("vad-audio-processor", VadAudioProcessor);

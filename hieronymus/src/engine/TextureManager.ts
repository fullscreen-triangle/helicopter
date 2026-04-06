export class TextureManager {
  private gl: WebGL2RenderingContext;
  private textures: WebGLTexture[] = [];
  private fbos: WebGLFramebuffer[] = [];

  constructor(gl: WebGL2RenderingContext) {
    this.gl = gl;
    // Enable float texture extensions
    gl.getExtension('EXT_color_buffer_float');
    gl.getExtension('OES_texture_float_linear');
  }

  createFloat32Texture(width: number, height: number): WebGLTexture {
    const gl = this.gl;
    const tex = gl.createTexture();
    if (!tex) throw new Error('Failed to create texture');

    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(
      gl.TEXTURE_2D, 0, gl.RGBA32F,
      width, height, 0,
      gl.RGBA, gl.FLOAT, null
    );
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_2D, null);

    this.textures.push(tex);
    return tex;
  }

  createFBO(texture: WebGLTexture): WebGLFramebuffer {
    const gl = this.gl;
    const fbo = gl.createFramebuffer();
    if (!fbo) throw new Error('Failed to create framebuffer');

    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D, texture, 0
    );

    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      throw new Error(`Framebuffer incomplete: ${status}`);
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.fbos.push(fbo);
    return fbo;
  }

  uploadImageData(
    texture: WebGLTexture,
    data: Uint8Array,
    width: number,
    height: number
  ): void {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(
      gl.TEXTURE_2D, 0, gl.RGBA,
      width, height, 0,
      gl.RGBA, gl.UNSIGNED_BYTE, data
    );
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  readPixels(
    fbo: WebGLFramebuffer,
    x: number,
    y: number,
    w: number,
    h: number
  ): Float32Array {
    const gl = this.gl;
    const buf = new Float32Array(w * h * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.readPixels(x, y, w, h, gl.RGBA, gl.FLOAT, buf);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return buf;
  }

  dispose(): void {
    const gl = this.gl;
    for (const tex of this.textures) {
      gl.deleteTexture(tex);
    }
    for (const fbo of this.fbos) {
      gl.deleteFramebuffer(fbo);
    }
    this.textures = [];
    this.fbos = [];
  }
}

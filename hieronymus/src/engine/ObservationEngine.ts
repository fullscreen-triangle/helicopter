import { ShaderCompiler } from './ShaderCompiler';
import { TextureManager } from './TextureManager';
import type { ObservationResult, MatchResult } from './types';

import vertexSrc from '@/shaders/vertex.glsl';
import encodeMicroscopySrc from '@/shaders/encode_microscopy.glsl';
import partitionSrc from '@/shaders/partition.glsl';
import interferenceSrc from '@/shaders/interference.glsl';
import entropySrc from '@/shaders/entropy.glsl';
import displaySrc from '@/shaders/display.glsl';

export class ObservationEngine {
  private gl: WebGL2RenderingContext;
  private compiler: ShaderCompiler;
  private textures: TextureManager;
  private programs: Map<string, WebGLProgram> = new Map();
  private fbos: WebGLFramebuffer[] = [];
  private texs: WebGLTexture[] = [];
  private imageTexture: WebGLTexture | null = null;
  private quadVAO: WebGLVertexArrayObject;
  private uniforms: Record<string, number> = {
    epsilon: 0.15,
    J: 1.0,
    beta: 2.0,
    nmax: 8,
    Aeg: 2.58,
    alpha: 0.5,
  };
  private width: number;
  private height: number;

  constructor(
    canvas: OffscreenCanvas | HTMLCanvasElement,
    width?: number,
    height?: number
  ) {
    const gl = canvas.getContext('webgl2', {
      alpha: false,
      antialias: false,
      preserveDrawingBuffer: true,
    }) as WebGL2RenderingContext | null;

    if (!gl) {
      throw new Error('WebGL2 is not available');
    }

    this.gl = gl;
    this.width = width || canvas.width;
    this.height = height || canvas.height;

    this.compiler = new ShaderCompiler(gl);
    this.textures = new TextureManager(gl);

    this.quadVAO = this.buildQuad();
    this.compileAllShaders();
    this.createPipelineTextures(this.width, this.height);
  }

  private buildQuad(): WebGLVertexArrayObject {
    const gl = this.gl;
    const verts = new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]);
    const buf = gl.createBuffer();
    const vao = gl.createVertexArray();

    if (!buf || !vao) throw new Error('Failed to create quad buffers');

    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);

    return vao;
  }

  private compileAllShaders(): void {
    const shaders: [string, string][] = [
      ['encode_microscopy', encodeMicroscopySrc],
      ['partition', partitionSrc],
      ['interference', interferenceSrc],
      ['entropy', entropySrc],
      ['display', displaySrc],
    ];

    for (const [name, fragSrc] of shaders) {
      try {
        const program = this.compiler.link(vertexSrc, fragSrc);
        this.programs.set(name, program);
      } catch (err) {
        throw new Error(`Failed to compile shader "${name}": ${(err as Error).message}`);
      }
    }
  }

  private createPipelineTextures(w: number, h: number): void {
    // Clean up old textures/fbos
    for (const tex of this.texs) {
      this.gl.deleteTexture(tex);
    }
    for (const fbo of this.fbos) {
      this.gl.deleteFramebuffer(fbo);
    }
    this.texs = [];
    this.fbos = [];

    // Create 4 float textures + FBOs for pipeline passes
    for (let i = 0; i < 4; i++) {
      const tex = this.textures.createFloat32Texture(w, h);
      const fbo = this.textures.createFBO(tex);
      this.texs.push(tex);
      this.fbos.push(fbo);
    }
  }

  private drawQuad(): void {
    const gl = this.gl;
    gl.bindVertexArray(this.quadVAO);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.bindVertexArray(null);
  }

  private bindTex(unit: number, tex: WebGLTexture): void {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, tex);
  }

  private setUniform(prog: WebGLProgram, name: string, value: number | number[]): void {
    const gl = this.gl;
    const loc = gl.getUniformLocation(prog, name);
    if (loc === null) return;
    if (typeof value === 'number') {
      if (Number.isInteger(value) && name.startsWith('u_pass') || name === 'u_tex') {
        gl.uniform1i(loc, value);
      } else {
        gl.uniform1f(loc, value);
      }
    }
  }

  private setUniformInt(prog: WebGLProgram, name: string, value: number): void {
    const gl = this.gl;
    const loc = gl.getUniformLocation(prog, name);
    if (loc !== null) gl.uniform1i(loc, value);
  }

  observe(
    imageData: Uint8Array,
    width: number,
    height: number,
    encoder: string
  ): ObservationResult {
    const gl = this.gl;
    const t0 = performance.now();
    const U = this.uniforms;

    // Resize pipeline if needed
    if (width !== this.width || height !== this.height) {
      this.width = width;
      this.height = height;
      this.createPipelineTextures(width, height);
    }

    // Upload image texture
    if (this.imageTexture) {
      gl.deleteTexture(this.imageTexture);
    }
    this.imageTexture = gl.createTexture();
    if (!this.imageTexture) throw new Error('Failed to create image texture');

    gl.bindTexture(gl.TEXTURE_2D, this.imageTexture);
    gl.texImage2D(
      gl.TEXTURE_2D, 0, gl.RGBA,
      width, height, 0,
      gl.RGBA, gl.UNSIGNED_BYTE, imageData
    );
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_2D, null);

    gl.viewport(0, 0, width, height);

    // Pass 0: Encode (microscopy)
    const encProg = this.programs.get(
      encoder === 'microscopy' ? 'encode_microscopy' : 'encode_microscopy'
    )!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbos[0]);
    gl.useProgram(encProg);
    this.bindTex(0, this.imageTexture);
    this.setUniformInt(encProg, 'u_image', 0);
    this.setUniform(encProg, 'u_nmax', U.nmax);
    this.drawQuad();

    // Pass 1: Partition (invisible pixel / ternary states)
    const partProg = this.programs.get('partition')!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbos[1]);
    gl.useProgram(partProg);
    this.bindTex(0, this.imageTexture);
    this.bindTex(1, this.texs[0]);
    this.setUniformInt(partProg, 'u_image', 0);
    this.setUniformInt(partProg, 'u_pass1', 1);
    this.setUniform(partProg, 'u_Aeg', U.Aeg * 1e-4);
    this.setUniform(partProg, 'u_nmax', U.nmax);
    this.drawQuad();

    // Pass 2: Interference / Stereogram consistency
    const intProg = this.programs.get('interference')!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbos[2]);
    gl.useProgram(intProg);
    this.bindTex(0, this.texs[0]);
    this.bindTex(1, this.texs[1]);
    this.setUniformInt(intProg, 'u_pass1', 0);
    this.setUniformInt(intProg, 'u_pass2', 1);
    this.setUniform(intProg, 'u_epsilon', U.epsilon);
    this.setUniform(intProg, 'u_J', U.J);
    this.setUniform(intProg, 'u_beta', U.beta);
    this.setUniform(intProg, 'u_nmax', U.nmax);
    this.drawQuad();

    // Pass 3: Entropy conservation
    const entProg = this.programs.get('entropy')!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbos[3]);
    gl.useProgram(entProg);
    this.bindTex(0, this.imageTexture);
    this.bindTex(1, this.texs[1]);
    this.bindTex(2, this.texs[2]);
    this.setUniformInt(entProg, 'u_image', 0);
    this.setUniformInt(entProg, 'u_pass2', 1);
    this.setUniformInt(entProg, 'u_pass3', 2);
    this.setUniform(entProg, 'u_alpha', U.alpha);
    this.setUniform(entProg, 'u_nmax', U.nmax);
    this.drawQuad();

    gl.flush();

    // Readback metrics from center sample region
    const sx = Math.max(0, Math.floor(width / 2) - 32);
    const sy = Math.max(0, Math.floor(height / 2) - 32);
    const sw = Math.min(64, width - sx);
    const sh = Math.min(64, height - sy);
    const N = sw * sh;

    // Read pass 2 (consistency) metrics
    const p3buf = this.textures.readPixels(this.fbos[2], sx, sy, sw, sh);

    // Read pass 3 (entropy) metrics
    const p4buf = this.textures.readPixels(this.fbos[3], sx, sy, sw, sh);

    // Read pass 0 (encode) for sharpness/noise
    const p1buf = this.textures.readPixels(this.fbos[0], sx, sy, sw, sh);

    let sumValid = 0, sumSk = 0, sumSt = 0, sumSe = 0, sumSum = 0;
    let sumN = 0, sumGrad = 0, sumVar = 0;

    for (let i = 0; i < N; i++) {
      sumValid += p3buf[i * 4 + 1]; // validation mask
      sumSk += p4buf[i * 4 + 0];
      sumSt += p4buf[i * 4 + 1];
      sumSe += p4buf[i * 4 + 2];
      sumSum += p4buf[i * 4 + 3];

      sumN += p1buf[i * 4 + 0]; // principal level
      sumGrad += p1buf[i * 4 + 1]; // angular level (gradient)
      const val = p1buf[i * 4 + 0];
      sumVar += val * val;
    }

    const meanN = sumN / N;
    const meanGrad = sumGrad / N;
    const variance = sumVar / N - meanN * meanN;
    const elapsed = performance.now() - t0;

    // Read partition texture from pass 0
    const partitionTexture = this.textures.readPixels(
      this.fbos[0], 0, 0, width, height
    );

    return {
      S_k: sumSk / N,
      S_t: sumSt / N,
      S_e: sumSe / N,
      conservation: sumSum / N,
      partitionDepth: meanN * U.nmax,
      sharpness: meanGrad * 4,
      noise: Math.sqrt(Math.max(variance, 0)),
      coherence: sumValid / N,
      visibility: sumValid / N,
      elapsed_ms: elapsed,
      partitionTexture,
    };
  }

  match(
    imgA: Uint8Array,
    imgB: Uint8Array,
    width: number,
    height: number
  ): MatchResult {
    const t0 = performance.now();

    // Observe both images
    const resultA = this.observe(imgA, width, height, 'microscopy');
    const resultB = this.observe(imgB, width, height, 'microscopy');

    // Compute S-distance on the entropy simplex
    const dSk = resultA.S_k - resultB.S_k;
    const dSt = resultA.S_t - resultB.S_t;
    const dSe = resultA.S_e - resultB.S_e;
    const S_distance = Math.sqrt(dSk * dSk + dSt * dSt + dSe * dSe);

    // Score = 1 - normalized distance
    const score = Math.max(0, 1 - S_distance / Math.sqrt(2));

    // Visibility = average coherence
    const visibility = (resultA.visibility + resultB.visibility) / 2;

    // Circuits = partition depth correlation
    const circuits = Math.abs(resultA.partitionDepth - resultB.partitionDepth);

    const elapsed = performance.now() - t0;

    return {
      score,
      visibility,
      circuits,
      S_distance,
      elapsed_ms: elapsed,
    };
  }

  setUniforms(u: Record<string, number>): void {
    this.uniforms = { ...this.uniforms, ...u };
  }

  dispose(): void {
    const gl = this.gl;
    this.textures.dispose();
    for (const prog of this.programs.values()) {
      gl.deleteProgram(prog);
    }
    this.programs.clear();
    if (this.imageTexture) {
      gl.deleteTexture(this.imageTexture);
    }
    if (this.quadVAO) {
      gl.deleteVertexArray(this.quadVAO);
    }
  }
}

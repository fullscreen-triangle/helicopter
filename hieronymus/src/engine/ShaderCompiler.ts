export class ShaderCompiler {
  private gl: WebGL2RenderingContext;

  constructor(gl: WebGL2RenderingContext) {
    this.gl = gl;
  }

  compile(source: string, type: number): WebGLShader {
    const gl = this.gl;
    const shader = gl.createShader(type);
    if (!shader) {
      throw new Error('Failed to create shader object');
    }
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const info = gl.getShaderInfoLog(shader) || 'Unknown error';
      gl.deleteShader(shader);
      throw new Error(`Shader compile error: ${info}`);
    }
    return shader;
  }

  link(vertSrc: string, fragSrc: string): WebGLProgram {
    const gl = this.gl;
    const program = gl.createProgram();
    if (!program) {
      throw new Error('Failed to create program object');
    }

    const vertShader = this.compile(vertSrc, gl.VERTEX_SHADER);
    const fragShader = this.compile(fragSrc, gl.FRAGMENT_SHADER);

    gl.attachShader(program, vertShader);
    gl.attachShader(program, fragShader);

    // Bind a_pos to location 0 before linking
    gl.bindAttribLocation(program, 0, 'a_pos');

    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      const info = gl.getProgramInfoLog(program) || 'Unknown error';
      gl.deleteProgram(program);
      gl.deleteShader(vertShader);
      gl.deleteShader(fragShader);
      throw new Error(`Program link error: ${info}`);
    }

    // Shaders can be detached after linking
    gl.detachShader(program, vertShader);
    gl.detachShader(program, fragShader);
    gl.deleteShader(vertShader);
    gl.deleteShader(fragShader);

    return program;
  }
}

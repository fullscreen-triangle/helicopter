#version 300 es
precision highp float;
uniform sampler2D u_image;
uniform float u_nmax;
in vec2 v_uv;
out vec4 fragColor;

float luminance(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

// Local power in 3x3 neighbourhood
float localPower(sampler2D tex, vec2 uv, vec2 ts) {
  float sum = 0.0;
  for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
      float v = luminance(texture(tex, uv + vec2(float(dx), float(dy)) * ts).rgb);
      sum += v * v;
    }
  }
  return sqrt(sum / 9.0);
}

// Shannon entropy from local 3x3 histogram (4 bins)
float localEntropy(sampler2D tex, vec2 uv, vec2 ts) {
  float bins[4];
  for (int i = 0; i < 4; i++) bins[i] = 0.0;
  for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
      float v = luminance(texture(tex, uv + vec2(float(dx), float(dy)) * ts).rgb);
      int b = int(clamp(v * 4.0, 0.0, 3.0));
      if      (b == 0) bins[0] += 1.0;
      else if (b == 1) bins[1] += 1.0;
      else if (b == 2) bins[2] += 1.0;
      else             bins[3] += 1.0;
    }
  }
  float H = 0.0;
  for (int i = 0; i < 4; i++) {
    float p = bins[i] / 9.0;
    if (p > 0.0) H -= p * log2(p);
  }
  return H / 2.0;
}

void main() {
  vec2 ts = 1.0 / vec2(textureSize(u_image, 0));

  // S_k: power in local frequency band
  float S_k = localPower(u_image, v_uv, ts);

  // S_t: temporal position (y-axis = time in spectrogram)
  float S_t = v_uv.y;

  // S_e: conservation constraint
  float S_e = clamp(1.0 - S_k - S_t, 0.0, 1.0);

  // Map to partition coordinates
  float H = localEntropy(u_image, v_uv, ts);
  float nF = clamp(H * u_nmax, 1.0, u_nmax);
  float n  = floor(nF);

  float il = luminance(texture(u_image, v_uv + vec2(-ts.x, 0.0)).rgb);
  float ir = luminance(texture(u_image, v_uv + vec2( ts.x, 0.0)).rgb);
  float ib = luminance(texture(u_image, v_uv + vec2(0.0, -ts.y)).rgb);
  float it = luminance(texture(u_image, v_uv + vec2(0.0,  ts.y)).rgb);
  float gx = (ir - il) * 0.5;
  float gy = (it - ib) * 0.5;
  float gradMag   = length(vec2(gx, gy));
  float gradAngle = atan(gy, gx);

  float ic = luminance(texture(u_image, v_uv).rgb);
  float laplacian = il + ir + ib + it - 4.0 * ic;

  float lF = clamp(gradMag * (n - 1.0) * 4.0, 0.0, n - 1.0);
  float l  = floor(lF);

  float mRaw  = floor(((gradAngle / 3.14159265) * 0.5 + 0.5) * (2.0 * l + 1.0));
  float m     = clamp(mRaw - l, -l, l);
  float mNorm = (m + l) / max(2.0 * l + 1.0, 1.0);

  float s = laplacian >= 0.0 ? 1.0 : 0.0;

  fragColor = vec4(
    n / u_nmax,
    l / u_nmax,
    mNorm,
    s
  );
}

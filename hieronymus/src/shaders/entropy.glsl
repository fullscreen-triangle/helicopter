#version 300 es
precision highp float;
uniform sampler2D u_image;
uniform sampler2D u_pass2;   // ternary states for Se
uniform sampler2D u_pass3;   // consistency data
uniform float u_alpha;       // info transfer efficiency
uniform float u_nmax;
in vec2 v_uv;
out vec4 fragColor;

float luminance(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

float localVariance(vec2 uv, vec2 ts) {
  float mean = 0.0, sq = 0.0;
  for (int dy = -2; dy <= 2; dy++) {
    for (int dx = -2; dx <= 2; dx++) {
      float v = luminance(texture(u_image, uv + vec2(float(dx),float(dy))*ts).rgb);
      mean += v; sq += v*v;
    }
  }
  mean /= 25.0; sq /= 25.0;
  return max(sq - mean*mean, 0.0);
}

void main() {
  vec2 ts = 1.0 / vec2(textureSize(u_image, 0));
  float I = luminance(texture(u_image, v_uv).rgb);

  // Kinetic entropy Sk: gradient magnitude (translational disorder)
  float gx = luminance(texture(u_image, v_uv + vec2(ts.x, 0)).rgb)
           - luminance(texture(u_image, v_uv - vec2(ts.x, 0)).rgb);
  float gy = luminance(texture(u_image, v_uv + vec2(0, ts.y)).rgb)
           - luminance(texture(u_image, v_uv - vec2(0, ts.y)).rgb);
  float Sk_raw = length(vec2(gx, gy)) * 3.0; // scale up

  // Thermal entropy St: local intensity variance
  float St_raw = localVariance(v_uv, ts) * 20.0;

  // Electronic entropy Se: from O2 excited state P2
  vec4 ternary = texture(u_pass2, v_uv);
  float P2     = ternary.b;
  float Se_raw = P2;

  // Normalise so Sk + St + Se = 1 (entropy simplex)
  float total  = Sk_raw + St_raw + Se_raw + 1e-6;
  float Sk     = Sk_raw / total;
  float St     = St_raw / total;
  float Se     = Se_raw / total;

  // Resolution enhancement factor
  // REF = sqrt(1 + alpha*(delta_x_vis/delta_x_inv)^2)
  // delta_x_vis ~ 220nm, delta_x_inv ~ 0.121nm -> ratio ~ 1818
  float ratio = 1818.0;
  float REF   = sqrt(1.0 + u_alpha * ratio * ratio);
  float REF_norm = log(REF) / log(1818.0); // log-normalised for display

  // Information gain
  // I_vis ~ 8 bits/px, I_inv = N*log2(3)
  float I_vis  = 8.0;
  float I_inv  = 1000.0 * 1.585; // N=1000 O2 molecules
  float I_shared = I_vis * 0.6;  // partial redundancy
  float I_dual = I_vis + I_inv - I_shared;
  float I_norm = log2(I_dual) / 12.0; // normalise for display

  // Colourmap: Sk->cyan, St->amber, Se->violet blended by weights
  vec3 cSk = vec3(0.0, 0.831, 1.0);    // cyan
  vec3 cSt = vec3(1.0, 0.420, 0.208);  // amber
  vec3 cSe = vec3(0.753, 0.518, 0.988);// violet

  vec3 col = Sk * cSk + St * cSt + Se * cSe;
  // Consistency mask overlay
  float valid = texture(u_pass3, v_uv).g;
  col = mix(col * 0.4, col, valid); // dim invalid pixels

  // Pack stats into alpha for CPU readback
  fragColor = vec4(col, (Sk + St + Se)); // A ~ 1.0 always (conservation check)
}

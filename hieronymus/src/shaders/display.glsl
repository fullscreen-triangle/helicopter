#version 300 es
precision highp float;
uniform sampler2D u_tex;
uniform int u_pass;
in vec2 v_uv;
out vec4 fragColor;

vec3 inferno(float t) {
  const vec3 c0 = vec3(0.0002, 0.0016, 0.0140);
  const vec3 c1 = vec3(0.1065, 0.0512, 0.3100);
  const vec3 c2 = vec3(0.3917, 0.0795, 0.5490);
  const vec3 c3 = vec3(0.6920, 0.1655, 0.4157);
  const vec3 c4 = vec3(0.9200, 0.3840, 0.1700);
  const vec3 c5 = vec3(0.9882, 0.9922, 0.7490);
  t = clamp(t, 0.0, 1.0);
  if (t < 0.25) return mix(c0, c1, t * 4.0);
  if (t < 0.50) return mix(c1, c2, (t - 0.25) * 4.0);
  if (t < 0.75) return mix(c2, c3, (t - 0.50) * 4.0);
  return mix(c3, t < 0.875 ? c4 : c5, (t - 0.75) * 4.0);
}

void main() {
  vec4 tex = texture(u_tex, v_uv);

  if (u_pass == 0) {
    // Pass 0: false-colour by principal level n (R channel)
    float n = tex.r;
    // Cyan-to-magenta heatmap
    vec3 c = mix(vec3(0.0,0.831,1.0), vec3(1.0,0.0,0.6), n);
    // Add angular level as brightness modulation
    c *= 0.5 + 0.5 * tex.g;
    fragColor = vec4(c, 1.0);
  }
  else if (u_pass == 1) {
    // Pass 1: P0=R(cyan), P1=G(neutral), P2=B(amber)
    vec3 cP0 = vec3(0.0, 0.831, 1.0);   // absorption -> cyan
    vec3 cP1 = vec3(0.2, 0.25,  0.35);  // ground -> dark
    vec3 cP2 = vec3(1.0, 0.420, 0.208); // emission -> amber
    vec3 c   = tex.r * cP0 + tex.g * cP1 + tex.b * cP2;
    fragColor = vec4(c, 1.0);
  }
  else if (u_pass == 2) {
    // Pass 2: disparity map with validation overlay
    float dcat  = tex.r;
    float valid = tex.g;
    float rho   = tex.b;
    vec3 disparity = inferno(dcat * 3.0);
    // Green tint for validated pixels
    disparity = mix(disparity, vec3(0.2,0.9,0.3) * 0.4, valid * 0.4);
    // Dim inconsistent regions
    disparity *= mix(0.3, 1.0, valid);
    fragColor = vec4(disparity, 1.0);
  }
  else {
    // Pass 3: entropy simplex colouring (already RGB)
    fragColor = vec4(tex.rgb, 1.0);
  }
}

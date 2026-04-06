#version 300 es
precision highp float;
uniform sampler2D u_image;
uniform sampler2D u_pass1;
uniform float u_Aeg;    // Einstein A coefficient (spontaneous emission)
uniform float u_nmax;
in vec2 v_uv;
out vec4 fragColor;

float luminance(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

void main() {
  float I     = luminance(texture(u_image, v_uv).rgb);
  float I0    = 1.0;  // reference intensity (normalised)

  // Beer-Lambert: optical density -> estimated [O2] (normalised 0-1)
  float OD     = -log(max(I, 0.001) / I0);
  float O2conc = clamp(OD / 2.0, 0.0, 1.0); // epsilon*d = 2.0 normalised

  // Local radiation energy density u(nu) proxy from image intensity
  float u_nu  = I * 0.5; // simplified: brighter = more photons

  // Einstein B coefficients (natural units, relative to Aeg)
  float Bge   = u_Aeg * 2.0;  // absorption B coefficient
  float Beg   = u_Aeg * 1.5;  // stimulated emission B coefficient

  // Steady-state solution to rate equations (dP/dt = 0):
  // P2 = Bge*u / (Aeg + Beg*u + Bge*u)  [excited -> emitting]
  float denom = u_Aeg + (Beg + Bge) * u_nu;
  float P2    = (Bge * u_nu) / max(denom, 1e-6);

  // P0 = Aeg*P2 / (Bge*u)  [ground absorbing]
  float P0    = (u_Aeg * P2) / max(Bge * u_nu, 1e-6);

  // Normalise so P0 + P1 + P2 = 1
  float Psum  = P0 + P2;
  P0 = clamp(P0 / max(Psum + 0.3, 1.0), 0.0, 1.0);
  P2 = clamp(P2 / max(Psum + 0.3, 1.0), 0.0, 1.0);
  float P1 = clamp(1.0 - P0 - P2, 0.0, 1.0);

  // Invisible partition signature: encode dominant ternary state
  // into (n,l,m,s) using O2 concentration as entropy proxy
  vec4 visSig = texture(u_pass1, v_uv);
  float nF    = clamp(O2conc * u_nmax, 1.0, u_nmax);
  float invN  = floor(nF) / u_nmax;

  // Encode invisible sig in A -- weighted average with visible n
  float invSig = mix(invN, visSig.r, 0.4); // partial channel independence

  fragColor = vec4(P0, P1, P2, invSig);
}

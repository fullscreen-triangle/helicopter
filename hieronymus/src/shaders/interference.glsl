#version 300 es
precision highp float;
uniform sampler2D u_pass1;  // visible partition (n,l,m,s)
uniform sampler2D u_pass2;  // invisible (P0,P1,P2,invSig)
uniform float u_epsilon;    // consistency threshold
uniform float u_J;          // coupling constant
uniform float u_beta;       // inverse temperature
uniform float u_nmax;
in vec2 v_uv;
out vec4 fragColor;

// Local inter-modality correlation over 5x5 neighbourhood
float interModalCorr(vec2 uv, vec2 ts) {
  float sumVis = 0.0, sumInv = 0.0;
  float sumVV = 0.0, sumII = 0.0, sumVI = 0.0;
  float N = 0.0;
  for (int dy = -2; dy <= 2; dy++) {
    for (int dx = -2; dx <= 2; dx++) {
      vec2 off = vec2(float(dx), float(dy)) * ts;
      float vis = texture(u_pass1, uv + off).r; // use n-component
      float inv = texture(u_pass2, uv + off).a; // invisible n-proxy
      sumVis += vis; sumInv += inv;
      sumVV  += vis * vis; sumII += inv * inv; sumVI += vis * inv;
      N += 1.0;
    }
  }
  float mV = sumVis / N, mI = sumInv / N;
  float stdV = sqrt(max(sumVV / N - mV * mV, 1e-8));
  float stdI = sqrt(max(sumII / N - mI * mI, 1e-8));
  return (sumVI / N - mV * mI) / (stdV * stdI);
}

void main() {
  vec2 ts = 1.0 / vec2(textureSize(u_pass1, 0));

  vec4 visSig = texture(u_pass1, v_uv); // (n,l,m,s) normalised
  vec4 invRaw = texture(u_pass2, v_uv); // (P0,P1,P2, invN)

  // Reconstruct invisible partition signature from ternary state
  // Map dominant state to pseudo-(n,l,m,s)
  float P0 = invRaw.r, P1 = invRaw.g, P2 = invRaw.b;
  float invN = invRaw.a;
  // l-proxy: P2 encodes angular complexity (emission = excitation)
  float invL = P2 * (visSig.r * u_nmax - 1.0) / u_nmax;
  // m-proxy: P0-P2 asymmetry
  float invM = (P2 - P0 + 1.0) * 0.5;
  // s-proxy: from P1 dominance
  float invS = step(0.5, P1);

  vec4 invSig = vec4(invN, invL, invM, invS);

  // Categorical distance d_cat in (n,l,m,s) space
  // Scale each component by its dynamic range
  vec4 diff   = (visSig - invSig) * vec4(1.0, 0.5, 0.5, 0.5);
  float dcat  = length(diff);

  // Inter-modality correlation rho
  float rho   = interModalCorr(v_uv, ts);

  // Cross-modal interaction potential V_cross = -J * rho * f(d_cat)
  float f_d   = exp(-dcat * 2.0); // decreasing function of disparity
  float Vcross = -u_J * rho * f_d;

  // Fusion weights (Boltzmann)
  float wVis  = exp(-u_beta * max(Vcross, 0.0));
  float wInv  = exp(-u_beta * max(-Vcross, 0.0));
  float wSum  = wVis + wInv + 1e-6;

  // Fused partition signature
  vec4 sigDual = (wVis * visSig + wInv * invSig) / wSum;

  // Validation mask M: 1 if consistent, 0 if not
  float valid = step(dcat, u_epsilon);

  // Output:
  // R: d_cat (disparity map -- key visualization)
  // G: validation mask
  // B: inter-modal correlation rho (mapped 0-1)
  // A: fused n-signature
  fragColor = vec4(
    clamp(dcat, 0.0, 1.0),
    valid,
    (rho + 1.0) * 0.5,
    sigDual.r
  );
}

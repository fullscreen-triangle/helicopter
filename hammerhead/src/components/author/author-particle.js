import "particles.js/particles";
import dynamic from "next/dynamic";
import React, { useEffect } from "react";

const MicroscopeScene = dynamic(
  () => import("../MicroscopeModel"),
  { ssr: false }
);

export default function AuthorDefault() {
  useEffect(() => {
    const particlesJS = window.particlesJS;
    particlesJS.load("particles-js", "particlesConfig.json", function () {
      console.log("particles loaded");
    });
  }, []);

  return (
    <>
      <div className="author_image">
        <div
          className="main"
          style={{ background: '#0a0e14' }}
        >
          <MicroscopeScene />
        </div>
        <div className="particle_wrapper">
          <div id="particles-js" />
        </div>
      </div>
    </>
  );
}

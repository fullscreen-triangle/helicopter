import Link from "next/link";
import React from "react";
import { LoadingTextAnimation } from "../AnimationText";

export default function HomeDefault({ ActiveIndex, handleOnClick }) {
  return (
    <>
      {/* <!-- HOME --> */}
      <div
        className={
          ActiveIndex === 0
            ? "cavani_tm_section active animated flipInX"
            : "cavani_tm_section active hidden animated flipOutX"
        }
        id="home_"
      >
        <div className="cavani_tm_home">
          <div className="content">
            <h3 className="name">Hammerhead</h3>
            <span className="line"></span>
            <p className="tagline" style={{marginBottom: '15px', fontSize: '14px'}}>The algorithm IS the microscope</p>
            <h3 className="job">
              <LoadingTextAnimation />
            </h3>
            <div className="cavani_tm_button transition_link">
              <Link href="#framework">
                <a onClick={() => handleOnClick(1)}>Explore Framework</a>
              </Link>
            </div>
          </div>
        </div>
      </div>
      {/* <!-- HOME --> */}
    </>
  );
}

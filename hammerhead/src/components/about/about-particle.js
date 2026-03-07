import React from 'react'
import ProgressBar from '../progressBar';
import { CircularProgressbar } from "react-circular-progressbar";
import "react-circular-progressbar/dist/styles.css";
import { Swiper, SwiperSlide } from 'swiper/react';
import 'swiper/css';

const circleProgressData = [
    { language: 'Compilation', progress: 99 },
    { language: 'Conservation', progress: 100 },
    { language: 'Generalization', progress: 95 },
];

const progressBarData = [
    { bgcolor: "#4a9eda", completed: 95, title: 'Resolution Enhancement' },
    { bgcolor: "#4a9eda", completed: 100, title: 'S-Entropy Conservation' },
    { bgcolor: "#4a9eda", completed: 90, title: 'Segmentation Accuracy' },
];

const keyResults = [
    {
        desc: "17.8x resolution enhancement achieved through categorical completion, where neural outputs match expert manual analysis exactly.",
        img: "img/panels/panel_chart_1_segmentation_resolution.png",
        info1: "Resolution Enhancement",
        info2: "Fluorescence Microscopy"
    },
    {
        desc: "Machine-precision S-entropy conservation with standard deviation 9.9e-17, proving information-theoretic correctness of the framework.",
        img: "img/panels/panel_chart_2_entropy_conservation.png",
        info1: "Conservation Proof",
        info2: "S-Entropy Coordinates"
    },
    {
        desc: "65ms average compilation time, representing a 27,496x speedup over expert manual pipeline construction.",
        img: "img/panels/panel_chart_3_speed_csp.png",
        info1: "Compilation Speed",
        info2: "Natural Language Interface"
    },
];

export default function AboutDefault({ ActiveIndex }) {
    return (
        <>
            {/* <!-- FRAMEWORK --> */}
            <div className={ActiveIndex === 1 ? "cavani_tm_section active animated flipInX" : "cavani_tm_section active hidden animated flipOutX"} id="about_">
            <div className="section_inner">
                    <div className="cavani_tm_about">
                        <div className="biography">
                            <div className="cavani_tm_title">
                                <span>The Framework</span>
                            </div>
                            <div className="wrapper">
                                <div className="left">
                                    <p><strong>Hammerhead</strong> is a categorical imaging framework that compiles natural language descriptions into type-safe morphism chains. Built on Partition Calculus, it enables users to describe imaging tasks in plain English and receive mathematically verified processing pipelines.</p>
                                    <p>The framework implements the <strong>observe &rarr; catalyze &rarr; fuse &rarr; access</strong> pipeline, where twelve independent measurement modalities reduce structural ambiguity from ~10<sup>60</sup> to unique determination through sequential exclusion.</p>
                                </div>
                                <div className="right">
                                    <ul>
                                        <li><span className="first">Framework:</span><span className="second">Partition Calculus</span></li>
                                        <li><span className="first">Language:</span><span className="second">Python 3.8+ / PyTorch</span></li>
                                        <li><span className="first">License:</span><span className="second">MIT</span></li>
                                        <li><span className="first">Compilation:</span><span className="second">65ms average</span></li>
                                        <li><span className="first">Resolution:</span><span className="second">17.8x enhancement</span></li>
                                        <li><span className="first">Repository:</span><span className="second"><a href="https://github.com/fullscreen-triangle/helicopter" target="_blank" rel="noopener noreferrer">GitHub</a></span></li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                        <div className="services">
                            <div className="wrapper">
                                <div className="service_list">
                                    <div className="cavani_tm_title">
                                        <span>Pipeline Stages</span>
                                    </div>
                                    <div className="list">
                                        <ul>
                                            <li>Observe (Image Acquisition)</li>
                                            <li>Catalyze (Feature Extraction)</li>
                                            <li>Fuse (Multi-modal Integration)</li>
                                            <li>Access (Result Retrieval)</li>
                                        </ul>
                                    </div>
                                </div>
                                <div className="service_list">
                                    <div className="cavani_tm_title">
                                        <span>Imaging Modalities</span>
                                    </div>
                                    <div className="list">
                                        <ul>
                                            <li>Fluorescence Microscopy</li>
                                            <li>Electron Microscopy</li>
                                            <li>Phase Contrast</li>
                                            <li>Spectral Multiplexing</li>
                                            <li>Virtual Imaging</li>
                                        </ul>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div className="skills">
                            <div className="wrapper">
                                <div className="programming">
                                    <div className="cavani_tm_title">
                                        <span>Performance Metrics</span>
                                    </div>
                                    <div className="cavani_progress">
                                        {progressBarData.map((item, idx) => (
                                            <ProgressBar key={idx} bgcolor={item.bgcolor} completed={item.completed} title={item.title} />
                                        ))}
                                    </div>
                                </div>
                                <div className="language">
                                    <div className="cavani_tm_title">
                                        <span>Framework Reliability</span>
                                    </div>
                                    <div className="circular_progress_bar">
                                        <div className='circle_holder'>
                                            {circleProgressData.map((item, idx) => (
                                                <div key={idx}>
                                                    <div className="list_inner">
                                                        <CircularProgressbar
                                                            value={item.progress}
                                                            text={`${item.progress}%`}
                                                            strokeWidth={3}
                                                            stroke='#4a9eda'
                                                            Language={item.language}
                                                            className={"list_inner"}
                                                        />
                                                        <div className="title"><span>{item.language}</span></div>
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div className="resume">
                            <div className="wrapper">
                                <div className="education">
                                    <div className="cavani_tm_title">
                                        <span>Theoretical Foundation</span>
                                    </div>
                                    <div className="list">
                                        <div className="univ">
                                            <ul>
                                                <li>
                                                    <div className="list_inner">
                                                        <div className="time">
                                                            <span>2024 - Present</span>
                                                        </div>
                                                        <div className="place">
                                                            <h3>Partition Calculus</h3>
                                                            <span>Categorical Imaging Framework</span>
                                                        </div>
                                                    </div>
                                                </li>
                                                <li>
                                                    <div className="list_inner">
                                                        <div className="time">
                                                            <span>2023 - 2024</span>
                                                        </div>
                                                        <div className="place">
                                                            <h3>Maxwell Demon Theory</h3>
                                                            <span>Pixel-Level Observation</span>
                                                        </div>
                                                    </div>
                                                </li>
                                                <li>
                                                    <div className="list_inner">
                                                        <div className="time">
                                                            <span>2022 - 2023</span>
                                                        </div>
                                                        <div className="place">
                                                            <h3>S-Entropy Coordinates</h3>
                                                            <span>Bounded Phase Space</span>
                                                        </div>
                                                    </div>
                                                </li>
                                            </ul>
                                        </div>
                                    </div>
                                </div>
                                <div className="experience">
                                    <div className="cavani_tm_title">
                                        <span>Publications</span>
                                    </div>
                                    <div className="list">
                                        <div className="univ">
                                            <ul>
                                                <li>
                                                    <div className="list_inner">
                                                        <div className="time">
                                                            <span>2025</span>
                                                        </div>
                                                        <div className="place">
                                                            <h3>Purpose-Driven Image Partition</h3>
                                                            <span>Neural Compilation Pipeline</span>
                                                        </div>
                                                    </div>
                                                </li>
                                                <li>
                                                    <div className="list_inner">
                                                        <div className="time">
                                                            <span>2025</span>
                                                        </div>
                                                        <div className="place">
                                                            <h3>Multi-Modal Image Analysis</h3>
                                                            <span>Fluorescence &amp; Electron Microscopy</span>
                                                        </div>
                                                    </div>
                                                </li>
                                                <li>
                                                    <div className="list_inner">
                                                        <div className="time">
                                                            <span>2024</span>
                                                        </div>
                                                        <div className="place">
                                                            <h3>Scattering-Puzzle Imaging</h3>
                                                            <span>Refraction Puzzle Reconstruction</span>
                                                        </div>
                                                    </div>
                                                </li>
                                            </ul>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div className="testimonials">
                            <div className="cavani_tm_title">
                                <span>Key Results</span>
                            </div>
                            <div className="list">
                                <ul className="">
                                    <li>
                                        <Swiper
                                            slidesPerView={1}
                                            spaceBetween={30}
                                            loop={true}
                                            className="custom-class"
                                            breakpoints={{
                                                768: {
                                                    slidesPerView: 2,
                                                }
                                            }}
                                        >
                                            {keyResults.map((item, i) => (
                                                <SwiperSlide key={i}>
                                                    <div className="list_inner">
                                                        <div className="text">
                                                            <i className="icon-quote-left" />
                                                            <p>{item.desc}</p>
                                                        </div>
                                                        <div className="details">
                                                            <div className="image">
                                                                <div className="main" data-img-url={item.img} />
                                                            </div>
                                                            <div className="info">
                                                                <h3>{item.info1}</h3>
                                                                <span>{item.info2}</span>
                                                            </div>
                                                        </div>
                                                    </div>
                                                </SwiperSlide>
                                            ))}
                                        </Swiper>
                                    </li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            {/* <!-- /FRAMEWORK --> */}
        </>
    )
}

import React, { useState } from 'react'
import Modal from 'react-modal';

const papers = [
    {
        title: "Purpose-Driven Image Partition Models",
        subtitle: "Neural Compilation",
        category: "life-sciences theory",
        img: "img/panels/panel_chart_1_segmentation_resolution.png",
        abstract: "We establish that the mapping from imaging task specifications to Partition Calculus morphism chains is a well-defined compilation problem, solvable by domain-specific language models trained through knowledge distillation. The neural compiler achieves 0.900 Dice segmentation, 17.8x resolution enhancement, and 65ms compilation time.",
        detail1: "The compilation function decomposes into four sequential phases: observe bridge (vision encoder to partition coordinates), catalyst selection (autoregressive decoder), cross-attention fusion (multi-modal combination), and amortized CSP solving (constraint satisfaction).",
        detail2: "Neurally-compiled morphism chains match expert-crafted chains exactly on all metrics, while reducing compilation time by 27,496x. S-entropy conservation holds to machine precision (std 9.9e-17).",
        year: "2025",
        status: "Submitted"
    },
    {
        title: "Multi-Modal Life Sciences Image Analysis",
        subtitle: "Fluorescence & Electron Microscopy",
        category: "life-sciences",
        img: "img/panels/panel_chart_4_generalization_fusion.png",
        abstract: "A unified categorical framework for analyzing life sciences images across fluorescence, electron, and phase contrast modalities. Cross-attention fusion recovers inter-modality correlation coefficients, enabling resolution enhancement beyond single-modality limits.",
        detail1: "The framework achieves zero generalization gap across biological targets including nuclei, cell membrane, mitochondria, and endoplasmic reticulum.",
        detail2: "S-entropy coordinates provide a bounded measure of information content that is conserved through all categorical operations, ensuring no information loss during multi-modal fusion.",
        year: "2025",
        status: "Submitted"
    },
    {
        title: "Scattering-Puzzle Refraction Imaging",
        subtitle: "Amortized CSP Solver",
        category: "theory",
        img: "img/panels/panel_chart_3_speed_csp.png",
        abstract: "An amortized neural constraint satisfaction solver replaces iterative reconstruction for refraction puzzle imaging. The solver exploits the scattering enhancement theorem: higher scattering strength increases transfer matrix rank, improving reconstruction quality.",
        detail1: "The neural CSP solver achieves 11.1 dB PSNR compared to 7.0 dB for iterative least-squares, with a scattering-rank correlation of 0.738 confirming the theoretical prediction.",
        detail2: "Single-pass neural inference replaces iterative optimization, enabling real-time reconstruction from scattered measurements.",
        year: "2024",
        status: "Published"
    },
    {
        title: "Partition Algebra for Categorical Observation",
        subtitle: "Foundational Theory",
        category: "theory",
        img: "img/panels/panel_chart_2_entropy_conservation.png",
        abstract: "The foundational theory of Partition Calculus establishing that image analysis is equivalent to categorical observation. Partition coordinates (n, l, m, s) with capacity C(n) = 2n^2 provide a complete description of observable structure.",
        detail1: "S-entropy conservation S_k + S_t + S_e = constant is proven as a fundamental invariant of categorical observation, analogous to energy conservation in physics.",
        detail2: "The twelve-catalyst vocabulary enables sequential exclusion of structural ambiguity, reducing the resolution from optical diffraction limits to sub-nanometer determination.",
        year: "2024",
        status: "Published"
    }
];

export default function PortfolioDefault({ ActiveIndex }) {
    const [activeTab, setActiveTab] = useState('all');
    const [isOpen, setIsOpen] = useState(false);
    const [modalContent, setModalContent] = useState({});

    const handleFilterClick = (filter) => {
        setActiveTab(filter);
    };

    const filteredPapers = activeTab === 'all'
        ? papers
        : papers.filter(p => p.category.includes(activeTab));

    return (
        <>
            {/* <!-- PAPERS --> */}
            <div className={ActiveIndex === 2 ? "cavani_tm_section active animated flipInX" : "cavani_tm_section hidden animated flipOutX"} id="portfolio_">
                <div className="section_inner">
                    <div className="cavani_tm_portfolio">
                        <div className="cavani_tm_title">
                            <span>Publications</span>
                        </div>

                        <div className="portfolio_filter">
                            <ul>
                                <li><a onClick={() => handleFilterClick('all')} href="#" className={activeTab === 'all' ? "current" : ""}>All</a></li>
                                <li><a onClick={() => handleFilterClick('life-sciences')} href="#" className={activeTab === 'life-sciences' ? "current" : ""}>Life Sciences</a></li>
                                <li><a onClick={() => handleFilterClick('theory')} href="#" className={activeTab === 'theory' ? "current" : ""}>Theory</a></li>
                            </ul>
                        </div>

                        <div className="portfolio_list">
                            <ul className="gallery_zoom">
                                {filteredPapers.map((paper, i) => (
                                    <li key={i} className="detail fadeInUp">
                                        <div className="list_inner">
                                            <div className="image">
                                                <img src="img/thumbs/1-1.jpg" alt="" />
                                                <div
                                                    className="main"
                                                    style={{ backgroundImage: `url(${paper.img})` }}
                                                    onClick={() => { setModalContent(paper); setIsOpen(true); }}
                                                ></div>
                                                <span className="icon"><i className="icon-doc-text-inv"></i></span>
                                                <div className="details">
                                                    <h3>{paper.title}</h3>
                                                    <span>{paper.subtitle}</span>
                                                </div>
                                            </div>
                                        </div>
                                    </li>
                                ))}
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
            {/* <!-- /PAPERS --> */}

            <Modal
                isOpen={isOpen}
                onRequestClose={() => setIsOpen(false)}
                contentLabel="Paper details"
                className="mymodal"
                overlayClassName="myoverlay"
                closeTimeoutMS={300}
                openTimeoutMS={300}
            >
                <div className="cavani_tm_modalbox opened">
                    <div className="box_inner">
                        <div className="close" onClick={() => setIsOpen(false)}>
                            <a href="#"><i className="icon-cancel"></i></a>
                        </div>
                        <div className="description_wrap">
                            <div className="popup_details">
                                <div className="top_image">
                                    <img src="img/thumbs/4-2.jpg" alt="" />
                                    <div className="main" style={{ backgroundImage: `url(${modalContent.img})` }} />
                                </div>
                                <div className="portfolio_main_title">
                                    <h3>{modalContent.title}</h3>
                                    <span>{modalContent.subtitle} | {modalContent.year} | {modalContent.status}</span>
                                </div>
                                <div className="main_details">
                                    <div className="textbox">
                                        <p>{modalContent.abstract}</p>
                                        <p>{modalContent.detail1}</p>
                                        <p>{modalContent.detail2}</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </Modal>
        </>
    )
}

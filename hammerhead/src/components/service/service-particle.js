import React, { useState } from 'react'
import { dataImage } from '../../plugin/plugin'
import Modal from 'react-modal';
import { SVG_Custom1, SVG_Custom3, SVG_Custom4, SVG_Custom6 } from '../../plugin/svg';

export default function Service({ ActiveIndex }) {
    const [isOpen7, setIsOpen7] = useState(false);
    const [modalContent, setModalContent] = useState({});

    function toggleModalFour() {
        setIsOpen7(!isOpen7);
    }
    const service = [
        {
            img: "img/panels/panel_2_resolution.png",
            svg: <SVG_Custom1 />,
            text: "17.8x resolution enhancement through categorical completion. Neural outputs match expert manual analysis across all biological targets.",
            title: "Super-Resolution",
            text1: "The framework achieves 17.8x resolution enhancement by applying categorical completion theory to microscopy images. Starting from a 200nm optical diffraction baseline, sequential catalyst application reduces effective resolution to 11.2nm.",
            text2: "Through dual-membrane pixel Maxwell demons, each pixel maintains conjugate states enabling zero-backaction observation. The exclusion factors of twelve independent catalysts multiply to determine the final resolution.",
            text3: "Results validated across fluorescence microscopy (BBBC039), electron microscopy, and virtual imaging modalities. Neural and expert selections achieve identical enhancement ratios."
        },
        {
            img: "img/panels/panel_3_s_entropy.png",
            svg: <SVG_Custom4 />,
            text: "Machine-precision S-entropy conservation (std 9.9e-17). Information-theoretic proof of zero information loss during categorical observation.",
            title: "Information Conservation",
            text1: "S-entropy coordinates (S_k, S_t, S_e) in [0,1]^3 provide a bounded measure of information content at each stage of the imaging pipeline. The conservation law S_k + S_t + S_e = constant is enforced at machine precision.",
            text2: "The conservation proof demonstrates that categorical observation preserves total information content. Unlike conventional image processing where information is irreversibly lost at each step, morphism chains transform information between components while preserving the total.",
            text3: "Standard deviation of 9.9e-17 across all test cases confirms machine-precision conservation. The entropy partitions as S_k : S_t : S_e = 0.919 : 0.081 : 10^-5 after full catalyst application."
        },
        {
            img: "img/panels/panel_4_speed.png",
            svg: <SVG_Custom3 />,
            text: "65ms average compilation from natural language to type-safe morphism chains. 27,496x speedup vs expert manual compilation.",
            title: "Neural Compilation",
            text1: "Users describe imaging tasks in plain English. The neural compiler parses the intent and generates a type-safe morphism chain through four sequential phases: observe bridge, catalyst selection, cross-attention fusion, and constraint satisfaction.",
            text2: "The categorical type system ensures pipeline correctness at compile time. Constrained decoding enforces catalyst compatibility, preventing incompatible combinations (e.g., conservation and dissipation catalysts cannot co-occur).",
            text3: "Average compilation time of 65ms enables interactive, real-time pipeline construction. The 27,496x speedup over expert manual compilation (30+ minutes) democratizes access to categorical imaging methods."
        },
        {
            img: "img/panels/panel_5_generalization.png",
            svg: <SVG_Custom6 />,
            text: "Zero generalization gap across biological targets. 0.900+ Dice segmentation on unseen structures including mitochondria and endoplasmic reticulum.",
            title: "Universal Generalization",
            text1: "The framework achieves identical performance on in-distribution and out-of-distribution biological targets, with a generalization gap of effectively zero across nuclei, cell membrane, mitochondria, and endoplasmic reticulum.",
            text2: "Categorical completion theory guarantees that the framework generalizes to any bounded phase space system. The compositional structure of morphism chains transfers across biological targets through shared catalyst types.",
            text3: "Curriculum training with progressive complexity builds transferable representations. The model learns to apply known catalyst types to novel targets by reasoning about shared biological properties rather than memorizing specific patterns."
        }
    ]
    return (
        <>
            {/* <!-- CAPABILITIES --> */}
            <div className={ActiveIndex === 5 ? "cavani_tm_section active animated flipInX" : "cavani_tm_section hidden animated flipOutX"} id="news_">
            <div className="section_inner">
                    <div className="cavani_tm_service">
                        <div className="cavani_tm_title">
                            <span>Capabilities</span>
                        </div>
                        <div className="service_list">
                            <ul>
                                {service.map((item, i) => (
                                    <li key={i}>
                                        <div className="list_inner" onClick={toggleModalFour}>
                                            {item.svg}
                                            <h3 className="title" onClick={toggleModalFour}>{item.title}</h3>
                                            <p className="text">{item.text}</p>
                                            <a className="cavani_tm_full_link" href="#" onClick={() => setModalContent(item)} />
                                            <img className="popup_service_image" src={item.img} alt="" />
                                            <div className="service_hidden_details">
                                                <div className="service_popup_informations">
                                                    <div className="descriptions">
                                                        <p>{item.text1}</p>
                                                        <p>{item.text2}</p>
                                                        <p>{item.text3}</p>
                                                    </div>
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
            {/* <!-- /CAPABILITIES --> */}

            {modalContent && (
                <Modal
                    isOpen={isOpen7}
                    onRequestClose={toggleModalFour}
                    contentLabel="My dialog"
                    className="mymodal"
                    overlayClassName="myoverlay"
                    closeTimeoutMS={300}
                    openTimeoutMS={300}
                >
                    <div className="cavani_tm_modalbox opened">
                        <div className="box_inner">
                            <div className="close" onClick={toggleModalFour} >
                                <a href="#"><i className="icon-cancel"></i></a>
                            </div>
                            <div className="description_wrap">
                                <div className="service_popup_informations">
                                    <div className="image">
                                        <img src="img/thumbs/4-2.jpg" alt="" />
                                        <div className="main" style={{ backgroundImage: `url(${modalContent.img})` }} />
                                    </div>
                                    <div className="details">
                                        <span>{modalContent.tag}</span>
                                        <h3>{modalContent.title}</h3>
                                    </div>
                                    <div className="descriptions">
                                        <p>{modalContent.text1}</p>
                                        <p>{modalContent.text2}</p>
                                        <p>{modalContent.text3}</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </Modal>
            )}
        </>
    )
}

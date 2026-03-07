import { Fragment, useEffect, useState } from "react";
import Modal from "react-modal";
const News = ({ ActiveIndex, animation }) => {
  const [isOpen4, setIsOpen4] = useState(false);
  const [modalContent, setModalContent] = useState({});

  useEffect(() => {
    var lists = document.querySelectorAll(".news_list > ul > li");
    let box = document.querySelector(".cavani_fn_moving_box");
    if (!box) {
      let body = document.querySelector("body");
      let div = document.createElement("div");
      div.classList.add("cavani_fn_moving_box");
      body.appendChild(div);
    }

    lists.forEach((list) => {
      list.addEventListener("mouseenter", (event) => {
        box.classList.add("opened");
        var imgURL = list.getAttribute("data-img");
        box.style.backgroundImage = `url(${imgURL})`;
        box.style.top = event.clientY - 50 + "px";
        if (imgURL === "") {
          box.classList.remove("opened");
          return false;
        }
      });
      list.addEventListener("mouseleave", () => {
        box.classList.remove("opened");
      });
    });
  }, []);

  function toggleModalFour(value) {
    setIsOpen4(!isOpen4);
    setModalContent(value);
  }
  const newsData = [
    {
      img: "img/panels/panel_chart_1_segmentation_resolution.png",
      tag: "Fluorescence",
      date: "2025",
      comments: "17.8x Enhancement",
      title: "Resolution Enhancement: Neural Matches Expert",
      text1:
        "Applied categorical completion to fluorescence microscopy images from the BBBC039 dataset. The neural compilation pipeline achieves 17.8x resolution enhancement, improving from a 200nm optical baseline to 11.2nm effective resolution.",
      text2:
        "Both neural and expert catalyst selection produce identical enhancement factors (ratio = 1.000), validating the knowledge distillation approach. The theoretical maximum of 266.7x represents perfect three-catalyst enhancement.",
      text3:
        "The resolution cascade applies conservation(dna_mass), phase_lock(chromatin), and thermal(metabolic) catalysts sequentially, each reducing structural ambiguity through exclusion factors.",
    },
    {
      img: "img/panels/panel_chart_2_entropy_conservation.png",
      tag: "Theory",
      date: "2025",
      comments: "std 9.9e-17",
      title: "S-Entropy Conservation to Machine Precision",
      text1:
        "S-entropy conservation S_k + S_t + S_e = 1 holds to IEEE 754 double-precision floating-point limits across all neurally-compiled morphism chains. Standard deviation: 9.9 x 10^-17.",
      text2:
        "The entropy partitions as S_k : S_t : S_e = 0.919 : 0.081 : 10^-5, confirming that the overwhelming majority of information resides in the known component after catalyst application.",
      text3:
        "Zero conservation violations across all test images. Maximum deviation from unity: 2.2 x 10^-16, at the level of machine epsilon.",
    },
    {
      img: "img/panels/panel_chart_3_speed_csp.png",
      tag: "Compilation",
      date: "2025",
      comments: "27,496x Speedup",
      title: "65ms Compilation: Three Orders of Magnitude Faster",
      text1:
        "The neural compiler achieves 65ms average compilation time from natural language task description to type-safe morphism chain. This represents a 27,496x speedup over expert manual compilation (30+ minutes).",
      text2:
        "Template matching approaches require 1-5 minutes and intermediate expertise. The neural compiler eliminates both the time and expertise requirements entirely.",
      text3:
        "Total pipeline time including execution is 854ms. The compilation phase (65ms) is dominated by the execution phase (789ms), suggesting that further speedups should target chain execution rather than compilation.",
    },
    {
      img: "img/panels/panel_1_segmentation.png",
      tag: "Segmentation",
      date: "2025",
      comments: "0.900 Dice",
      title: "Segmentation Across Methods and Targets",
      text1:
        "Nuclear segmentation on BBBC039 achieves 0.900 Dice score for both expert-crafted and neurally-compiled morphism chains. Classical methods (Otsu: 0.948, Watershed: 0.949) achieve higher Dice on well-separated nuclei.",
      text2:
        "The morphism chain approach trades raw segmentation accuracy for compositional guarantees: type safety, S-entropy conservation, and catalyst compatibility that classical thresholding cannot provide.",
      text3:
        "The precision-recall tradeoff reveals that morphism chains favor recall (0.881) over precision (0.658), reflecting over-segmentation at sub-diffraction boundaries in dense nuclear regions.",
    },
    {
      img: "img/panels/panel_5_generalization.png",
      tag: "Generalization",
      date: "2025",
      comments: "Zero Gap",
      title: "Zero Generalization Gap Across Biological Targets",
      text1:
        "Neural and expert performance are effectively identical across all four biological targets: nuclei (0.900), cell membrane (0.900), mitochondria (0.907), and endoplasmic reticulum (0.900).",
      text2:
        "Out-of-distribution targets (mitochondria, ER) perform comparably to in-distribution targets (nuclei, membrane), with mitochondria achieving slightly higher Dice due to distinct morphological signatures.",
      text3:
        "The zero generalization gap validates that curriculum training successfully transfers compositional structure across biological targets, as predicted by the theoretical framework.",
    },
    {
      img: "img/panels/panel_6_csp.png",
      tag: "CSP Solver",
      date: "2025",
      comments: "11.1 dB PSNR",
      title: "Amortized CSP Solver Outperforms Iterative Methods",
      text1:
        "The neural constraint satisfaction solver achieves 11.1 dB PSNR compared to 7.0 dB for iterative least-squares reconstruction, a 4.1 dB improvement through single-pass inference.",
      text2:
        "The scattering enhancement theorem is confirmed: higher scattering strength increases transfer matrix rank (correlation 0.738), improving reconstruction quality rather than degrading it.",
      text3:
        "Variable ordering via the neural network correlates with the most-constrained-first heuristic, validating the learned constraint propagation strategy.",
    },
  ];
  return (
    <Fragment>
      <div
        className={
          ActiveIndex === 3
            ? `cavani_tm_section active animated ${animation ? animation : "fadeInUp"
            }`
            : "cavani_tm_section hidden animated"
        }
        id="news__"
      >
        <div className="section_inner">
          <div className="cavani_tm_news">
            <div className="cavani_tm_title">
              <span>Validation Results</span>
            </div>
            <div className="news_list">
              <ul>
                {newsData.map((news, i) => (
                  <li data-img={news.img} key={i}>
                    <div className="list_inner">
                      <span className="number">{`${i <= 9 ? 0 : ""}${i + 1
                        }`}</span>
                      <div className="details">
                        <div className="extra_metas">
                          <ul>
                            <li>
                              <span>{news.date}</span>
                            </li>
                            <li>
                              <span>
                                <a
                                  href="#"
                                  onClick={() => toggleModalFour(news)}
                                >
                                  {news.tag}
                                </a>
                              </span>
                            </li>
                            <li>
                              <span>
                                <a
                                  href="#"
                                  onClick={() => toggleModalFour(news)}
                                >
                                  {news.comments}
                                </a>
                              </span>
                            </li>
                          </ul>
                        </div>
                        <div className="post_title">
                          <h3>
                            <a href="#" onClick={() => toggleModalFour(news)}>
                              {news.title}
                            </a>
                          </h3>
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
      {modalContent && (
        <Modal
          isOpen={isOpen4}
          onRequestClose={toggleModalFour}
          contentLabel="My dialog"
          className="mymodal"
          overlayClassName="myoverlay"
          closeTimeoutMS={300}
          openTimeoutMS={300}
        >
          <div className="cavani_tm_modalbox opened">
            <div className="box_inner">
              <div className="close" onClick={toggleModalFour}>
                <a href="#">
                  <i className="icon-cancel"></i>
                </a>
              </div>
              <div className="description_wrap">
                <div className="news_popup_informations">
                  <div className="image">
                    <img src="img/thumbs/4-2.jpg" alt="" />
                    <div
                      className="main"
                      style={{ backgroundImage: `url(${modalContent.img})` }}
                    />
                  </div>
                  <div className="details">
                    <div className="meta">
                      <ul>
                        <li><span>{modalContent.date}</span></li>
                        <li><span><a href="#">{modalContent.tag}</a></span></li>
                        <li><span><a href="#">{modalContent.comments}</a></span></li>
                      </ul>
                    </div>
                    <div className="title">
                      <h3>{modalContent.title}</h3>
                    </div>
                  </div>
                  <div className="text">
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
    </Fragment>
  );
};
export default News;

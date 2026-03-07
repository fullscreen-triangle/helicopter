import React, { useState, useEffect } from 'react'
import MagicCursor from '../../layout/magic-cursor';
import { customCursor } from '../../plugin/plugin';

export default function ContactDefault({ ActiveIndex }) {
    const [trigger, setTrigger] = useState(false);
    useEffect(() => {
        customCursor();
    });

    const [form, setForm] = useState({ email: "", name: "", msg: "" });
    const [active, setActive] = useState(null);
    const [error, setError] = useState(false);
    const [success, setSuccess] = useState(false);
    const onChange = (e) => {
        setForm({ ...form, [e.target.name]: e.target.value });
    };
    const { email, name, msg } = form;
    const onSubmit = (e) => {
        e.preventDefault();
        if (email && name && msg) {
            setSuccess(true);
            setTimeout(() => {
                setForm({ email: "", name: "", msg: "" });
                setSuccess(false);
            }, 2000);
        } else {
            setError(true);
            setTimeout(() => {
                setError(false);
            }, 2000);
        }
    };
    return (
        <>
            {/* <!-- COLLABORATE --> */}
            <div className={ActiveIndex === 4 ? "cavani_tm_section active animated flipInX" : "cavani_tm_section hidden animated flipOutX"} id="contact_">
            <div className="section_inner">
                    <div className="cavani_tm_contact">
                        <div className="cavani_tm_title">
                            <span>Collaborate</span>
                        </div>
                        <div className="short_info">
                            <ul>
                                <li>
                                    <div className="list_inner">
                                        <i className="icon-location"></i>
                                        <span>Open Source - MIT License</span>
                                    </div>
                                </li>
                                <li>
                                    <div className="list_inner">
                                        <i className="icon-mail-3"></i>
                                        <span><a href="https://github.com/fullscreen-triangle/helicopter" target="_blank" rel="noopener noreferrer">GitHub Repository</a></span>
                                    </div>
                                </li>
                                <li>
                                    <div className="list_inner">
                                        <i className="icon-doc-text-inv"></i>
                                        <span>Funding &amp; Collaboration Welcome</span>
                                    </div>
                                </li>
                            </ul>
                        </div>
                        <div className="form">
                            <div className="left">
                                <div className="fields">
                                    {/* Contact Form */}
                                    <form className="contact_form" onSubmit={(e) => onSubmit(e)}>
                                        <div
                                            className="returnmessage"
                                            data-success="Thank you for your interest. We will respond soon."
                                            style={{ display: success ? "block" : "none" }}
                                        >
                                            <span className="contact_success">
                                                Thank you for your interest. We will respond soon.
                                            </span>
                                        </div>
                                        <div
                                            className="empty_notice"
                                            style={{ display: error ? "block" : "none" }}
                                        >
                                            <span>Please Fill Required Fields!</span>
                                        </div>

                                        <div className="fields">
                                            <ul>
                                                <li
                                                    className={`input_wrapper ${active === "name" || name ? "active" : ""
                                                        }`}
                                                >
                                                    <input
                                                        onFocus={() => setActive("name")}
                                                        onBlur={() => setActive(null)}
                                                        onChange={(e) => onChange(e)}
                                                        value={name}
                                                        name="name"
                                                        id="name"
                                                        type="text"
                                                        placeholder="Name"
                                                    />
                                                </li>
                                                <li
                                                    className={`input_wrapper ${active === "email" || email ? "active" : ""
                                                        }`}
                                                >
                                                    <input
                                                        onFocus={() => setActive("email")}
                                                        onBlur={() => setActive(null)}
                                                        onChange={(e) => onChange(e)}
                                                        value={email}
                                                        name="email"
                                                        id="email"
                                                        type="email"
                                                        placeholder="Email"
                                                    />
                                                </li>
                                                <li
                                                    className={`last ${active === "message" || msg ? "active" : ""
                                                        }`}
                                                >
                                                    <textarea
                                                        onFocus={() => setActive("message")}
                                                        onBlur={() => setActive(null)}
                                                        name="msg"
                                                        onChange={(e) => onChange(e)}
                                                        value={msg}
                                                        id="message"
                                                        placeholder="Describe your interest or collaboration proposal"
                                                    />
                                                </li>
                                            </ul>
                                            <div className="cavani_tm_button">
                                                <input
                                                    className='a'
                                                    type="submit"
                                                    id="send_message"
                                                    value="Send Inquiry"
                                                />
                                            </div>
                                        </div>
                                    </form>
                                    {/* /Contact Form */}
                                </div>
                            </div>
                            <div className="right">
                                <div className="map_wrap">
                                    <div className="demo-cta-box" style={{ padding: '40px', textAlign: 'center', borderRadius: '8px', height: '375px', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center' }}>
                                        <h3 style={{ marginBottom: '15px', fontSize: '20px' }}>Try Hammerhead</h3>
                                        <p style={{ marginBottom: '10px', fontSize: '14px', lineHeight: '1.6' }}>Describe your imaging task in natural language and receive a type-safe morphism chain in 65ms.</p>
                                        <p style={{ marginBottom: '25px', fontSize: '13px' }}>Interactive demo coming soon.</p>
                                        <div className="cavani_tm_button">
                                            <a href="https://github.com/fullscreen-triangle/helicopter" target="_blank" rel="noopener noreferrer">View on GitHub</a>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            {/* <!-- /COLLABORATE --> */}
        </>
    )
}

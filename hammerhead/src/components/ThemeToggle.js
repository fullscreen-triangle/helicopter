import React, { useState, useEffect } from 'react';

export default function ThemeToggle() {
    const [isDark, setIsDark] = useState(true);

    useEffect(() => {
        const saved = localStorage.getItem('hammerhead-theme');
        const dark = saved ? saved === 'dark' : true;
        setIsDark(dark);
        if (dark) {
            document.body.classList.add('dark');
        } else {
            document.body.classList.remove('dark');
        }
    }, []);

    const toggle = () => {
        const next = !isDark;
        setIsDark(next);
        localStorage.setItem('hammerhead-theme', next ? 'dark' : 'light');
        if (next) {
            document.body.classList.add('dark');
        } else {
            document.body.classList.remove('dark');
        }
    };

    return (
        <button className="theme-toggle" onClick={toggle} aria-label="Toggle theme">
            {isDark ? '☀️' : '🌙'}
        </button>
    );
}

'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import React, { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import { ChevronDown } from 'lucide-react';

const primaryLinks = [
  { href: '/observe/microscopy', title: 'Observe' },
  { href: '/match',              title: 'Match' },
  { href: '/publications',       title: 'Publications' },
  { href: '/docs',               title: 'Docs' },
  { href: '/about',              title: 'About' },
];

const toolLinks = [
  { href: '/tools/analysis-studio', title: 'Sandbox' },
  { href: '/tools/scope-playground', title: 'Playground' },
  { href: '/tools/mic-demo',         title: 'MIC Demo' },
];

const allLinks = [...primaryLinks, ...toolLinks];

function NavLink({ href, title, active }: { href: string; title: string; active: boolean }) {
  return (
    <Link href={href} className="relative group text-sm tracking-wider whitespace-nowrap">
      <span className={active ? 'text-cyan-400' : 'text-light'}>{title}</span>
      <span
        className={`inline-block h-[1px] bg-cyan-400 absolute left-0 -bottom-0.5
          group-hover:w-full transition-[width] ease duration-300
          ${active ? 'w-full' : 'w-0'}`}
      />
    </Link>
  );
}

function ToolsDropdown({ pathname }: { pathname: string }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const anyActive = toolLinks.some(l => pathname === l.href);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen(o => !o)}
        className={`flex items-center gap-1 text-sm tracking-wider ${anyActive ? 'text-cyan-400' : 'text-light'} hover:text-cyan-400 transition-colors`}
      >
        Tools <ChevronDown size={13} className={`transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>

      {open && (
        <div className="absolute right-0 top-full mt-2 w-44 bg-[#0c1118] border border-gray-800 rounded-lg shadow-xl z-50 overflow-hidden">
          {toolLinks.map(link => (
            <Link
              key={link.href}
              href={link.href}
              onClick={() => setOpen(false)}
              className={`block px-4 py-2.5 text-sm tracking-wider transition-colors
                ${pathname === link.href
                  ? 'text-cyan-400 bg-cyan-400/10'
                  : 'text-gray-400 hover:text-light hover:bg-white/5'}`}
            >
              {link.title}
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}

export default function NavbarApp() {
  const pathname = usePathname();
  const [isOpen, setIsOpen] = useState(false);

  return (
    <header className="w-full flex items-center justify-between px-12 py-5 font-medium z-20 text-light border-b border-gray-800/50 md:px-6 sm:px-4">
      <Link href="/" className="flex items-center gap-2 shrink-0">
        <span className="text-cyan-400 font-bold text-lg tracking-wider">HIERONYMUS</span>
        <span className="text-gray-600 text-xs tracking-widest sm:hidden">OBSERVATION PLATFORM</span>
      </Link>

      {/* Desktop nav — visible on md and above */}
      <nav className="flex items-center gap-6 md:hidden">
        {primaryLinks.map(link => (
          <NavLink key={link.href} href={link.href} title={link.title} active={pathname === link.href} />
        ))}
        <ToolsDropdown pathname={pathname} />
      </nav>

      {/* Mobile hamburger — only on small screens */}
      <button
        type="button"
        className="hidden md:flex flex-col items-center justify-center gap-1.5 p-1"
        onClick={() => setIsOpen(!isOpen)}
        aria-label="Toggle menu"
      >
        <span className={`bg-light block h-0.5 w-6 rounded-sm transition-all duration-300 ${isOpen ? 'rotate-45 translate-y-2' : ''}`} />
        <span className={`bg-light block h-0.5 w-6 rounded-sm transition-all duration-300 ${isOpen ? 'opacity-0' : ''}`} />
        <span className={`bg-light block h-0.5 w-6 rounded-sm transition-all duration-300 ${isOpen ? '-rotate-45 -translate-y-2' : ''}`} />
      </button>

      {/* Mobile menu overlay */}
      {isOpen && (
        <motion.div
          className="fixed inset-0 bg-dark/95 z-50 flex flex-col items-center justify-center gap-6 backdrop-blur-md"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
        >
          <button
            className="absolute top-5 right-6 text-gray-400 text-2xl"
            onClick={() => setIsOpen(false)}
          >
            ✕
          </button>
          {allLinks.map(link => (
            <button
              key={link.href}
              className={`text-lg tracking-wider ${pathname === link.href ? 'text-cyan-400' : 'text-gray-300'}`}
              onClick={() => { setIsOpen(false); window.location.href = link.href; }}
            >
              {link.title}
            </button>
          ))}
        </motion.div>
      )}
    </header>
  );
}

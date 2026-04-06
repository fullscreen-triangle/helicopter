'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import React, { useState } from 'react';
import { motion } from 'framer-motion';

const navLinks = [
  { href: '/', title: 'Home' },
  { href: '/observe', title: 'Observe' },
  { href: '/observe/microscopy', title: 'Microscopy' },
  { href: '/publications', title: 'Publications' },
  { href: '/about', title: 'About' },
];

function NavLink({
  href,
  title,
  active,
  className = '',
}: {
  href: string;
  title: string;
  active: boolean;
  className?: string;
}) {
  return (
    <Link
      href={href}
      className={`${className} rounded relative group text-sm tracking-wider`}
    >
      {title}
      <span
        className={`inline-block h-[1px] bg-light absolute left-0 -bottom-0.5
          group-hover:w-full transition-[width] ease duration-300
          ${active ? 'w-full' : 'w-0'}`}
      >
        &nbsp;
      </span>
    </Link>
  );
}

export default function NavbarApp() {
  const pathname = usePathname();
  const [isOpen, setIsOpen] = useState(false);

  return (
    <header className="w-full flex items-center justify-between px-12 py-5 font-medium z-10 text-light border-b border-gray-800/50 sm:px-6">
      <Link href="/" className="flex items-center gap-2">
        <span className="text-cyan-400 font-bold text-lg tracking-wider">
          HIERONYMUS
        </span>
        <span className="text-gray-600 text-xs tracking-widest hidden sm:hidden md:inline">
          OBSERVATION PLATFORM
        </span>
      </Link>

      {/* Desktop nav */}
      <nav className="flex items-center gap-6 lg:hidden">
        {navLinks.map((link) => (
          <NavLink
            key={link.href}
            href={link.href}
            title={link.title}
            active={pathname === link.href}
          />
        ))}
      </nav>

      {/* Mobile toggle */}
      <button
        type="button"
        className="flex-col items-center justify-center hidden lg:flex"
        onClick={() => setIsOpen(!isOpen)}
        aria-label="Toggle menu"
      >
        <span
          className={`bg-light block h-0.5 w-6 rounded-sm transition-all duration-300 ease-out ${
            isOpen ? 'rotate-45 translate-y-1' : '-translate-y-0.5'
          }`}
        />
        <span
          className={`bg-light block h-0.5 w-6 rounded-sm transition-all duration-300 ease-out ${
            isOpen ? 'opacity-0' : 'opacity-100'
          } my-0.5`}
        />
        <span
          className={`bg-light block h-0.5 w-6 rounded-sm transition-all duration-300 ease-out ${
            isOpen ? '-rotate-45 -translate-y-1' : 'translate-y-0.5'
          }`}
        />
      </button>

      {/* Mobile menu */}
      {isOpen && (
        <motion.div
          className="min-w-[70vw] flex justify-between items-center flex-col fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 py-16 bg-dark/95 rounded-lg z-50 backdrop-blur-md border border-gray-800"
          initial={{ scale: 0, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
        >
          <nav className="flex items-center justify-center flex-col gap-4">
            {navLinks.map((link) => (
              <button
                key={link.href}
                className={`text-sm tracking-wider ${
                  pathname === link.href ? 'text-cyan-400' : 'text-gray-400'
                }`}
                onClick={() => {
                  setIsOpen(false);
                  window.location.href = link.href;
                }}
              >
                {link.title}
              </button>
            ))}
          </nav>
        </motion.div>
      )}
    </header>
  );
}

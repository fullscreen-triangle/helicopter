'use client';

import { motion } from 'framer-motion';

export default function AboutPage() {
  return (
    <div className="min-h-[calc(100vh-140px)] flex flex-col items-center justify-center px-8 py-16">
      <motion.h1
        className="text-4xl font-bold mb-4"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        About Hieronymus
      </motion.h1>
      <motion.div
        className="text-gray-500 max-w-2xl text-center space-y-4"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2 }}
      >
        <p>
          Hieronymus is a client-side GPU observation platform that transforms
          scientific images into their thermodynamic coordinates using WebGL2
          fragment shaders.
        </p>
        <p>
          Every pixel is encoded into partition coordinates (n, l, m, s) and
          validated against the entropy conservation law: S_k + S_t + S_e = 1.
        </p>
        <p>
          No data leaves your browser. All computation happens on your GPU.
        </p>
      </motion.div>
    </div>
  );
}

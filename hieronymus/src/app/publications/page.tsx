'use client';

import { motion } from 'framer-motion';

export default function PublicationsPage() {
  return (
    <div className="min-h-[calc(100vh-140px)] flex flex-col items-center justify-center px-8 py-16">
      <motion.h1
        className="text-4xl font-bold mb-4"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        Publications
      </motion.h1>
      <motion.p
        className="text-gray-500 max-w-md text-center"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2 }}
      >
        Research papers underpinning the observation architecture. Coming soon.
      </motion.p>
    </div>
  );
}

"use client";

import { motion } from "framer-motion";
import GlitchText from "@/components/GlitchText";

export default function Slide14Questions() {
  return (
    <div className="relative flex-1 flex flex-col items-center justify-center overflow-hidden">
      
      <div className="relative z-10 text-center">
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
        >
          <h1 className="text-9xl font-display font-black mb-8">
            <GlitchText 
              text="Questions?" 
              className="text-neon-secondary text-glow-magenta"
            />
          </h1>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.5, duration: 0.6 }}
          className="space-y-4"
        >
          <p className="text-3xl text-neon-primary font-bold">
            CIC-IIoT-2025 Security Analysis
          </p>
          <p className="text-xl text-neon-text">
            Machine Learning for Intrusion Detection
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 2.2 }}
          className="mt-16 flex items-center justify-center gap-4"
        >
          <span className="text-neon-accent text-lg">ML Security</span>
          <span className="text-neon-muted text-2xl">•</span>
          <span className="text-neon-primary text-lg">EPITA SCIA 2026</span>
        </motion.div>
      </div>

      <motion.div
        className="absolute bottom-0 left-0 right-0 h-48 bg-gradient-to-t from-neon-bg to-transparent"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
      />

      <motion.div
        className="absolute top-0 left-0 right-0 h-48 bg-gradient-to-b from-neon-bg to-transparent"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
      />
    </div>
  );
}

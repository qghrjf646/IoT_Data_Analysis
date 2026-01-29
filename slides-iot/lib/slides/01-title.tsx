"use client";

import { motion } from "framer-motion";
import GlitchText from "@/components/GlitchText";
import { TEAM_MEMBERS } from "@/lib/slides-config";

export default function Slide01Title() {
  return (
    <div className="relative flex-1 flex flex-col items-center justify-center overflow-hidden">
      
      <div className="relative z-10 text-center">
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
        >
          <h1 className="text-6xl md:text-8xl font-display font-black mb-4 leading-tight">
            <GlitchText 
              text="CIC-IIoT-2025" 
              className="text-neon-primary text-glow-cyan"
            />
          </h1>
        </motion.div>

        <motion.h2
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.2, duration: 0.6 }}
          className="text-4xl text-neon-text font-mono mb-4"
        >
          Security Analysis
        </motion.h2>

        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.5, duration: 0.6 }}
          className="text-2xl text-neon-muted mb-16"
        >
          Machine Learning for Intrusion Detection in Industrial IoT
        </motion.p>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 2 }}
          className="flex flex-wrap justify-center gap-4 max-w-4xl mx-auto"
        >
          {TEAM_MEMBERS.map((member, index) => (
            <motion.div
              key={member}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 2.2 + index * 0.15 }}
              className="px-4 py-2 bg-neon-bg/60 backdrop-blur-sm border border-neon-primary/30 rounded-lg whitespace-nowrap"
            >
              <span className="text-neon-text text-sm font-mono">{member}</span>
            </motion.div>
          ))}
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 3 }}
          className="mt-12 flex items-center justify-center gap-4"
        >
          <span className="text-neon-accent text-sm">EPITA</span>
          <span className="text-neon-muted">•</span>
          <span className="text-neon-secondary text-sm">ML Security</span>
          <span className="text-neon-muted">•</span>
          <span className="text-neon-primary text-sm">SCIA 2026</span>
        </motion.div>
      </div>

      <motion.div
        className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-neon-bg to-transparent"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
      />
    </div>
  );
}

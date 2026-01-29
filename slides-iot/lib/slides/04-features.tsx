"use client";

import { motion } from "framer-motion";
import NeonCard from "@/components/NeonCard";
import Image from "next/image";

export default function Slide04Features() {
  const features = [
    { name: "network_mss_max", corr: 0.526 },
    { name: "network_mss_avg", corr: 0.525 },
    { name: "network_header-length_min", corr: 0.464 },
    { name: "network_protocols_dst_count", corr: 0.423 },
    { name: "network_packets_all_count", corr: 0.367 },
  ];

  return (
    <div className="relative flex-1 flex flex-col p-8 overflow-hidden">

      <motion.h2
        initial={{ opacity: 0, x: -50 }}
        animate={{ opacity: 1, x: 0 }}
        className="relative z-10 text-7xl font-display font-bold text-neon-primary mb-2"
      >
        Key Discriminative Features
      </motion.h2>
      
      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
        className="relative z-10 text-xl text-neon-muted mb-8"
      >
        Features les plus corrélées avec les attaques
      </motion.p>

      <div className="relative z-10 flex-1 grid grid-cols-2 gap-8">
        <div className="flex flex-col gap-6">
          <NeonCard delay={0.4} glow="cyan" title="Top Correlated Features">
            <div className="space-y-3 mt-4">
              {features.map((feat, i) => (
                <motion.div
                  key={feat.name}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.6 + i * 0.1 }}
                  className="flex justify-between items-center py-2 border-b border-neon-muted/20"
                >
                  <span className="text-neon-text font-mono text-sm">{feat.name}</span>
                  <div className="flex items-center gap-3">
                    <div className="w-24 h-2 bg-neon-bg/50 rounded-full overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${feat.corr * 100}%` }}
                        transition={{ delay: 0.8 + i * 0.1, duration: 0.8 }}
                        className="h-full bg-neon-primary rounded-full"
                      />
                    </div>
                    <span className="text-neon-primary font-mono font-bold w-16 text-right">
                      {feat.corr.toFixed(3)}
                    </span>
                  </div>
                </motion.div>
              ))}
            </div>
          </NeonCard>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.2 }}
            className="p-4 bg-neon-bg/60 border border-neon-accent/30 rounded-xl"
          >
            <p className="text-neon-text">
              <span className="text-neon-accent font-bold">TCP MSS</span> et la diversité des protocoles 
              sont de forts indicateurs d&apos;attaque
            </p>
          </motion.div>
        </div>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.8 }}
          className="rounded-xl overflow-hidden border-2 border-neon-primary/30 bg-neon-bg/40 flex items-center justify-center p-4"
        >
          <img
            src="/figures/correlation_heatmap.png"
            alt="Correlation Heatmap"
            className="max-w-full max-h-full object-contain"
          />
        </motion.div>
      </div>
    </div>
  );
}

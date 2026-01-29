"use client";

import { motion } from "framer-motion";
import NeonCard from "@/components/NeonCard";
import AnimatedBar from "@/components/AnimatedBar";
import Image from "next/image";

export default function Slide06AnomalyResults() {
  const results = [
    { model: "Isolation Forest", f1: 0.812, auprc: 0.860, mcc: 0.694, winner: false },
    { model: "One-Class SVM", f1: 0.789, auprc: 0.826, mcc: 0.663, winner: false },
    { model: "LOF", f1: 0.831, auprc: 0.873, mcc: 0.721, winner: true },
  ];

  return (
    <div className="relative flex-1 flex flex-col p-8 overflow-hidden">

      <motion.h2
        initial={{ opacity: 0, x: -50 }}
        animate={{ opacity: 1, x: 0 }}
        className="relative z-10 text-7xl font-display font-bold text-neon-secondary mb-2"
      >
        Anomaly Detection Results
      </motion.h2>
      
      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
        className="relative z-10 text-xl text-neon-muted mb-8"
      >
        Comparaison des performances
      </motion.p>

      <div className="relative z-10 flex-1 grid grid-cols-2 gap-8">
        <div className="flex flex-col gap-4">
          <NeonCard delay={0.4} glow="magenta" title="Results">
            <div className="mt-4">
              <table className="w-full">
                <thead>
                  <tr className="text-neon-muted text-sm border-b border-neon-muted/20">
                    <th className="text-left py-2">Model</th>
                    <th className="text-right py-2">F1</th>
                    <th className="text-right py-2">AUPRC</th>
                    <th className="text-right py-2">MCC</th>
                  </tr>
                </thead>
                <tbody>
                  {results.map((r, i) => (
                    <motion.tr
                      key={r.model}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 0.6 + i * 0.15 }}
                      className={`border-b border-neon-muted/10 ${r.winner ? "bg-neon-secondary/10" : ""}`}
                    >
                      <td className={`py-3 ${r.winner ? "text-neon-secondary font-bold" : "text-neon-text"}`}>
                        {r.winner && "★ "}{r.model}
                      </td>
                      <td className="py-3 text-right font-mono text-neon-primary">{r.f1.toFixed(3)}</td>
                      <td className="py-3 text-right font-mono text-neon-primary">{r.auprc.toFixed(3)}</td>
                      <td className="py-3 text-right font-mono text-neon-primary">{r.mcc.toFixed(3)}</td>
                    </motion.tr>
                  ))}
                </tbody>
              </table>
            </div>
          </NeonCard>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.2 }}
            className="p-6 bg-neon-secondary/10 border border-neon-secondary/30 rounded-xl"
          >
            <h4 className="text-neon-secondary font-bold text-xl mb-2">Winner: Local Outlier Factor</h4>
            <p className="text-neon-muted">Density-based methods excel on this dataset</p>
          </motion.div>
        </div>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.8 }}
          className="rounded-xl overflow-hidden border-2 border-neon-secondary/30 bg-neon-bg/40 flex items-center justify-center p-4"
        >
          <img
            src="/figures/anomaly_detection_comparison.png"
            alt="Anomaly Detection Comparison"
            className="max-w-full max-h-full object-contain"
          />
        </motion.div>
      </div>
    </div>
  );
}

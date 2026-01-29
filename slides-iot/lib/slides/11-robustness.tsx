"use client";

import { motion } from "framer-motion";
import NeonCard from "@/components/NeonCard";

export default function Slide11Robustness() {
  const results = [
    { model: "Linear SVM", astute: "90.2%", robust: "3.1%", ratio: "3.5%", winner: false },
    { model: "Gradient Boosting", astute: "94.4%", robust: "34.2%", ratio: "36.2%", winner: false },
    { model: "Random Forest", astute: "94.6%", robust: "41.8%", ratio: "44.2%", winner: true },
  ];

  return (
    <div className="relative flex-1 flex flex-col p-8 overflow-hidden">

      <motion.h2
        initial={{ opacity: 0, x: -50 }}
        animate={{ opacity: 1, x: 0 }}
        className="relative z-10 text-5xl font-display font-bold text-neon-accent mb-2 leading-tight"
      >
        Model Robustness Comparison
      </motion.h2>
      
      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
        className="relative z-10 text-xl text-neon-muted mb-8"
      >
        ε = 0.5 FGSM Attack
      </motion.p>

      <div className="relative z-10 flex-1 flex flex-col justify-center">
        <NeonCard delay={0.4} glow="violet" className="max-w-5xl mx-auto w-full">
          <div className="overflow-x-auto custom-scrollbar">
            <table className="w-full min-w-[600px]">
              <thead>
                <tr className="text-neon-muted border-b border-neon-muted/20">
                  <th className="text-left py-4 text-lg whitespace-nowrap">Model</th>
                  <th className="text-right py-4 text-lg whitespace-nowrap">Astute Acc.</th>
                  <th className="text-right py-4 text-lg whitespace-nowrap">Robust Acc.</th>
                  <th className="text-right py-4 text-lg whitespace-nowrap">Robustness Ratio</th>
                </tr>
              </thead>
            <tbody>
              {results.map((r, i) => (
                <motion.tr
                  key={r.model}
                  initial={{ opacity: 0, x: -30 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.6 + i * 0.2 }}
                  className={`border-b border-neon-muted/10 ${r.winner ? "bg-neon-accent/10" : ""}`}
                >
                  <td className={`py-4 text-lg ${r.winner ? "text-neon-accent font-bold" : "text-neon-text"}`}>
                    {r.winner && "★ "}{r.model}
                  </td>
                  <td className="py-4 text-right font-mono text-neon-primary text-lg">{r.astute}</td>
                  <td className="py-4 text-right font-mono text-neon-secondary text-lg">{r.robust}</td>
                  <td className={`py-4 text-right font-mono text-lg font-bold ${r.winner ? "text-neon-accent" : "text-neon-text"}`}>
                    {r.ratio}
                  </td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        </NeonCard>

        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.4 }}
          className="mt-12 max-w-4xl mx-auto text-center"
        >
          <div className="p-6 bg-neon-accent/10 border border-neon-accent/30 rounded-xl">
            <p className="text-2xl text-neon-text">
              <span className="text-neon-accent font-bold">Finding:</span> Random Forest retains{" "}
              <span className="text-neon-primary font-bold">44%</span> accuracy under FGSM attack
            </p>
            <p className="text-neon-muted mt-2">
              Linear models collapse to near-random performance
            </p>
          </div>
        </motion.div>
      </div>
    </div>
  );
}

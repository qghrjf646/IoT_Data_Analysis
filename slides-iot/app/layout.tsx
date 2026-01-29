import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "CIC-IIoT-2025 Security Analysis",
  description: "Machine Learning for Intrusion Detection in Industrial IoT",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}

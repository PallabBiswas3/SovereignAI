import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SovereignAI Workbench",
  description: "Air-gapped agentic AI for confidential enterprise work",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}


import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Chat with Vansh Narang",
  description: "AI persona of Vansh Narang — ask about his background or schedule a call",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="h-full">{children}</body>
    </html>
  );
}

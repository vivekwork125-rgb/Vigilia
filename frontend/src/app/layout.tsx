import type { Metadata } from "next";
import "./globals.css";
export const metadata: Metadata = {
  title: "VIGILIA — Evidence-grounded intelligence",
  description:
    "Move from a question to source evidence. Search, investigate, and verify surveillance events.",
};
export default function Layout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

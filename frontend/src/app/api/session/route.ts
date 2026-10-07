import { NextRequest, NextResponse } from "next/server";
import { timingSafeEqual } from "node:crypto";
export async function POST(request: NextRequest) {
  const { token } = await request.json();
  const expected = process.env.API_TOKEN || "";
  const a = Buffer.from(typeof token === "string" ? token : "");
  const b = Buffer.from(expected);
  if (!expected || a.length !== b.length || !timingSafeEqual(a, b))
    return NextResponse.json(
      { detail: "Invalid access token" },
      { status: 401 },
    );
  const response = NextResponse.json({ status: "authenticated" });
  response.cookies.set("vigilia_access", expected, {
    httpOnly: true,
    sameSite: "strict",
    secure: process.env.NODE_ENV === "production",
    maxAge: 28800,
    path: "/",
  });
  return response;
}

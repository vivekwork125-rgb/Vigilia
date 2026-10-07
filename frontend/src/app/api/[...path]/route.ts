import { NextRequest } from "next/server";
export const runtime = "nodejs";
export const dynamic = "force-dynamic";
async function proxy(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> },
) {
  const token = process.env.API_TOKEN;
  if (token && request.cookies.get("vigilia_access")?.value !== token)
    return Response.json({ detail: "Access token required" }, { status: 401 });
  const origin = request.headers.get("origin");
  if (
    request.method !== "GET" &&
    origin &&
    origin !== request.nextUrl.origin &&
    new URL(origin).host !== request.headers.get("host")
  )
    return Response.json(
      { detail: "Cross-origin write denied" },
      { status: 403 },
    );
  const { path } = await context.params;
  const headers = new Headers();
  for (const key of ["content-type", "range", "if-range"]) {
    const value = request.headers.get(key);
    if (value) headers.set(key, value);
  }
  if (token) headers.set("authorization", `Bearer ${token}`);
  try {
    const upstream = await fetch(
      `${process.env.BACKEND_URL || "http://127.0.0.1:8100"}/${path.map(encodeURIComponent).join("/")}${request.nextUrl.search}`,
      {
        method: request.method,
        headers,
        body: request.method === "GET" ? undefined : request.body,
        // Node streaming uploads avoid buffering entire surveillance videos in memory.
        duplex: "half",
        cache: "no-store",
        signal: request.signal,
      } as RequestInit & { duplex: string },
    );
    const responseHeaders = new Headers();
    for (const key of [
      "content-type",
      "content-length",
      "content-range",
      "accept-ranges",
      "content-disposition",
    ]) {
      const value = upstream.headers.get(key);
      if (value) responseHeaders.set(key, value);
    }
    responseHeaders.set("cache-control", "private, no-store");
    responseHeaders.set("x-content-type-options", "nosniff");
    return new Response(upstream.body, {
      status: upstream.status,
      headers: responseHeaders,
    });
  } catch {
    return Response.json(
      {
        detail: "Evidence API is unavailable. Start the backend on port 8100.",
      },
      { status: 502 },
    );
  }
}
export const GET = proxy;
export const POST = proxy;

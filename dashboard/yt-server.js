import http from 'http';
import { execSync, spawn } from 'child_process';
import { existsSync, mkdirSync, readdirSync, statSync, createReadStream } from 'fs';
/* global process */
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DOWNLOADS = path.join(__dirname, 'public', 'downloads');
if (!existsSync(DOWNLOADS)) mkdirSync(DOWNLOADS, { recursive: true });

// Check yt-dlp exists
try {
  execSync('which yt-dlp', { stdio: 'ignore' });
  console.log('✅ yt-dlp found');
} catch {
  console.error('❌ yt-dlp not found. Install with: brew install yt-dlp');
  process.exit(1);
}

const server = http.createServer((req, res) => {
  // CORS
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') { res.writeHead(204); res.end(); return; }

  // POST /download — download a YouTube video
  if (req.method === 'POST' && req.url === '/download') {
    let body = '';
    req.on('data', c => body += c);
    req.on('end', () => {
      try {
        const { url } = JSON.parse(body);
        if (!url) {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'No URL provided' }));
          return;
        }

        console.log(`⬇ Downloading: ${url}`);
        const outTemplate = path.join(DOWNLOADS, '%(id)s.%(ext)s');

        const proc = spawn('yt-dlp', [
          '-f', 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best',
          '--merge-output-format', 'mp4',
          '-o', outTemplate,
          '--no-playlist',
          '--restrict-filenames',
          url,
        ]);

        let stderr = '';
        proc.stderr.on('data', d => { stderr += d.toString(); });
        proc.stdout.on('data', d => { process.stdout.write(d); });

        proc.on('close', (code) => {
          if (code !== 0) {
            console.error(`yt-dlp failed:`, stderr);
            res.writeHead(500, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: 'Download failed', details: stderr }));
            return;
          }

          // Find the most recently modified mp4
          const files = readdirSync(DOWNLOADS).filter(f => f.endsWith('.mp4'));
          const sorted = files.sort((a, b) => {
            return statSync(path.join(DOWNLOADS, b)).mtimeMs -
                   statSync(path.join(DOWNLOADS, a)).mtimeMs;
          });

          if (sorted.length === 0) {
            res.writeHead(500, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: 'No MP4 found after download' }));
            return;
          }

          // Use relative path — Vite serves public/downloads/ at same origin
          const videoUrl = `/downloads/${sorted[0]}`;
          console.log(`✅ Ready: ${videoUrl}`);
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ videoUrl, file: sorted[0] }));
        });
      } catch (e) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: e.message }));
      }
    });
    return;
  }

  // GET /downloads/* — serve video files with range support
  if (req.method === 'GET' && req.url.startsWith('/downloads/')) {
    const filename = decodeURIComponent(req.url.replace('/downloads/', ''));
    const filePath = path.join(DOWNLOADS, filename);
    if (!existsSync(filePath)) {
      res.writeHead(404);
      res.end('Not found');
      return;
    }
    const stat = statSync(filePath);
    const range = req.headers.range;

    if (range) {
      const parts = range.replace(/bytes=/, '').split('-');
      const start = parseInt(parts[0], 10);
      const end = parts[1] ? parseInt(parts[1], 10) : stat.size - 1;
      res.writeHead(206, {
        'Content-Range': `bytes ${start}-${end}/${stat.size}`,
        'Accept-Ranges': 'bytes',
        'Content-Length': end - start + 1,
        'Content-Type': 'video/mp4',
      });
      createReadStream(filePath, { start, end }).pipe(res);
    } else {
      res.writeHead(200, {
        'Content-Length': stat.size,
        'Content-Type': 'video/mp4',
        'Accept-Ranges': 'bytes',
      });
      createReadStream(filePath).pipe(res);
    }
    return;
  }

  res.writeHead(404);
  res.end('Not found');
});

const PORT = 3001;
server.listen(PORT, () => {
  console.log(`🎬 YouTube download server on http://localhost:${PORT}`);
  console.log(`   POST /download { "url": "https://youtube.com/watch?v=..." }`);
});

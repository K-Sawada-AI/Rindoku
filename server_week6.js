const http = require('http');
const fs = require('fs');
const path = require('path');
const root = __dirname;
const port = Number(process.env.PORT || 8000);
const mime = {'.html':'text/html; charset=utf-8','.md':'text/markdown; charset=utf-8','.css':'text/css; charset=utf-8','.js':'text/javascript; charset=utf-8','.png':'image/png','.jpg':'image/jpeg','.jpeg':'image/jpeg','.svg':'image/svg+xml','.json':'application/json'};
const server = http.createServer((req,res) => {
  let pathname;
  try { pathname = decodeURIComponent(new URL(req.url,'http://localhost').pathname); }
  catch {res.writeHead(400);res.end('Bad Request');return;}
  let filename=path.resolve(root, '.'+pathname);
  if(filename!==root && !filename.startsWith(root+path.sep)) {res.writeHead(403);res.end('Forbidden');return;}
  fs.stat(filename,(err,stat)=>{
    if(err){res.writeHead(404);res.end('Not found');return;}
    if(stat.isDirectory()) filename=path.join(filename,'week6.html');
    fs.readFile(filename,(err,data)=>{
      if(err){res.writeHead(404);res.end('Not found');return;}
      res.writeHead(200,{'Content-Type':mime[path.extname(filename).toLowerCase()]||'application/octet-stream','Cache-Control':'no-store'});
      res.end(data);
    });
  });
});
server.listen(port,()=>console.log(`Week 6: http://localhost:${port}/week6.html`));

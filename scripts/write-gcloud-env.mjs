import fs from 'fs'

const raw = fs.readFileSync('.env', 'utf8')
const line = raw.split(/\r?\n/).find(l => /^\s*GEMINI_API_KEY\s*=/.test(l))
if (!line) throw new Error('GEMINI_API_KEY not found in .env')

let v = line.replace(/^\s*GEMINI_API_KEY\s*=\s*/, '').trim()
if ((v.startsWith('"') && v.endsWith('"')) || (v.startsWith("'") && v.endsWith("'"))) {
  v = v.slice(1, -1)
}

fs.writeFileSync('.gcloud-env.yaml', `GEMINI_API_KEY: ${JSON.stringify(v)}\n`)
console.log('Wrote .gcloud-env.yaml')

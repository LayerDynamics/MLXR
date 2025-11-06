/**
 * Markdown rendering and utilities for MLXR
 * Provides safe markdown rendering with syntax highlighting
 */

import { marked } from 'marked'
import DOMPurify from 'dompurify'

/**
 * Regular expressions for markdown parsing
 *
 * HEADING_PATTERN: Matches markdown headings (# through ######)
 * - Uses [^\r\n]+ instead of .+ to prevent ReDoS (exponential backtracking)
 * - Safe for use with untrusted user input
 * - Pattern without 'g' flag to avoid shared state issues
 */
const HEADING_PATTERN = /^(#{1,6})\s+([^\r\n]+)$/m

/**
 * Create a new heading regex with global flag for iteration
 * Returns a fresh regex instance to avoid lastIndex state issues
 */
const createHeadingRegex = (): RegExp => new RegExp(HEADING_PATTERN.source, 'gm')

/**
 * Markdown rendering options
 */
export interface MarkdownOptions {
  sanitize?: boolean
  breaks?: boolean
  gfm?: boolean
  highlight?: (code: string, lang: string) => string
  baseUrl?: string
}

/**
 * Configure marked with default options
 */
function configureMarked(options: MarkdownOptions = {}) {
  marked.setOptions({
    breaks: options.breaks ?? true,
    gfm: options.gfm ?? true,
    ...(options.highlight && { highlight: options.highlight }),
    ...(options.baseUrl && { baseUrl: options.baseUrl }),
  })
}

/**
 * Render markdown to HTML
 * @param markdown - Markdown string to render
 * @param options - Rendering options
 * @returns Rendered and sanitized HTML
 */
export function renderMarkdown(
  markdown: string,
  options: MarkdownOptions = {}
): string {
  configureMarked(options)

  // Render markdown
  const html = marked.parse(markdown) as string

  // Sanitize HTML to prevent XSS
  if (options.sanitize !== false) {
    return DOMPurify.sanitize(html, {
      ALLOWED_TAGS: [
        'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
        'p', 'br', 'hr',
        'strong', 'em', 'u', 's', 'mark', 'code',
        'ul', 'ol', 'li',
        'blockquote', 'pre',
        'a', 'img',
        'table', 'thead', 'tbody', 'tr', 'th', 'td',
        'div', 'span',
      ],
      ALLOWED_ATTR: [
        'href', 'src', 'alt', 'title',
        'class', 'id',
        'data-*',
      ],
      ALLOW_DATA_ATTR: true,
    })
  }

  return html
}

/**
 * Render markdown to plain text (strip all HTML)
 * @param markdown - Markdown string
 * @returns Plain text
 */
export function markdownToPlainText(markdown: string): string {
  const html = marked.parse(markdown) as string
  const div = document.createElement('div')
  div.innerHTML = html
  return div.textContent || div.innerText || ''
}

/**
 * Extract code blocks from markdown
 * @param markdown - Markdown string
 * @returns Array of code blocks with language and content
 */
export function extractCodeBlocks(markdown: string): Array<{
  language: string
  code: string
  index: number
}> {
  const codeBlockRegex = /```(\w*)\n([\s\S]*?)```/g
  const blocks: Array<{ language: string; code: string; index: number }> = []
  let match: RegExpExecArray | null

  while ((match = codeBlockRegex.exec(markdown)) !== null) {
    blocks.push({
      language: match[1] || 'text',
      code: match[2].trim(),
      index: match.index,
    })
  }

  return blocks
}

/**
 * Extract inline code from markdown
 * @param markdown - Markdown string
 * @returns Array of inline code snippets
 */
export function extractInlineCode(markdown: string): string[] {
  const inlineCodeRegex = /`([^`]+)`/g
  const codes: string[] = []
  let match: RegExpExecArray | null

  while ((match = inlineCodeRegex.exec(markdown)) !== null) {
    codes.push(match[1])
  }

  return codes
}

/**
 * Extract links from markdown
 * @param markdown - Markdown string
 * @returns Array of links with text and URL
 */
export function extractLinks(markdown: string): Array<{
  text: string
  url: string
  title?: string
}> {
  const linkRegex = /\[([^\]]+)\]\(([^)]+?)(?:\s+"([^"]+)")?\)/g
  const links: Array<{ text: string; url: string; title?: string }> = []
  let match: RegExpExecArray | null

  while ((match = linkRegex.exec(markdown)) !== null) {
    links.push({
      text: match[1],
      url: match[2],
      title: match[3],
    })
  }

  return links
}

/**
 * Extract images from markdown
 * @param markdown - Markdown string
 * @returns Array of images with alt text and URL
 */
export function extractImages(markdown: string): Array<{
  alt: string
  url: string
  title?: string
}> {
  const imageRegex = /!\[([^\]]*)\]\(([^)]+?)(?:\s+"([^"]+)")?\)/g
  const images: Array<{ alt: string; url: string; title?: string }> = []
  let match: RegExpExecArray | null

  while ((match = imageRegex.exec(markdown)) !== null) {
    images.push({
      alt: match[1],
      url: match[2],
      title: match[3],
    })
  }

  return images
}

/**
 * Extract headings from markdown
 * @param markdown - Markdown string
 * @returns Array of headings with level and text
 */
export function extractHeadings(markdown: string): Array<{
  level: number
  text: string
  id: string
}> {
  const headingRegex = createHeadingRegex()
  const headings: Array<{ level: number; text: string; id: string }> = []
  let match: RegExpExecArray | null

  while ((match = headingRegex.exec(markdown)) !== null) {
    const level = match[1].length
    const text = match[2].trim()
    const id = text
      .toLowerCase()
      .replace(/[^\w\s-]/g, '')
      .replace(/\s+/g, '-')

    headings.push({ level, text, id })
  }

  return headings
}

/**
 * Generate table of contents from markdown
 * @param markdown - Markdown string
 * @returns HTML string for table of contents
 */
export function generateTableOfContents(markdown: string): string {
  const headings = extractHeadings(markdown)

  if (headings.length === 0) {
    return ''
  }

  let toc = '<nav class="toc"><ul>'
  let currentLevel = headings[0].level

  headings.forEach((heading) => {
    if (heading.level > currentLevel) {
      toc += '<ul>'.repeat(heading.level - currentLevel)
    } else if (heading.level < currentLevel) {
      toc += '</ul>'.repeat(currentLevel - heading.level)
    }

    toc += `<li><a href="#${heading.id}">${heading.text}</a></li>`
    currentLevel = heading.level
  })

  // Close remaining lists
  toc += '</ul>'.repeat(currentLevel - headings[0].level + 1)
  toc += '</nav>'

  return toc
}

/**
 * Escape markdown special characters
 * @param text - Text to escape
 * @returns Escaped text
 */
export function escapeMarkdown(text: string): string {
  return text.replace(/([\\`*_{}[\]()#+\-.!|])/g, '\\$1')
}

/**
 * Unescape markdown special characters
 * @param text - Text to unescape
 * @returns Unescaped text
 */
export function unescapeMarkdown(text: string): string {
  return text.replace(/\\([\\`*_{}[\]()#+\-.!|])/g, '$1')
}

/**
 * Truncate markdown text to specified length
 * @param markdown - Markdown string
 * @param length - Maximum length
 * @param suffix - Suffix to add (default: '...')
 * @returns Truncated markdown
 */
export function truncateMarkdown(
  markdown: string,
  length: number,
  suffix = '...'
): string {
  const plainText = markdownToPlainText(markdown)

  if (plainText.length <= length) {
    return markdown
  }

  // Find the position to truncate in the original markdown
  let charCount = 0
  let truncatePos = 0

  for (let i = 0; i < markdown.length; i++) {
    if (markdown[i] !== '[' && markdown[i] !== ']' && markdown[i] !== '(' && markdown[i] !== ')') {
      charCount++
    }

    if (charCount >= length - suffix.length) {
      truncatePos = i
      break
    }
  }

  return markdown.slice(0, truncatePos) + suffix
}

/**
 * Convert markdown to formatted plain text with basic formatting
 * @param markdown - Markdown string
 * @returns Formatted plain text
 */
export function markdownToFormattedText(markdown: string): string {
  let text = markdown

  // Convert headings (uses shared HEADING_PATTERN to keep regexes in sync)
  text = text.replace(createHeadingRegex(), '\n$2\n')

  // Convert bold
  text = text.replace(/\*\*([^*]+)\*\*/g, '$1')
  text = text.replace(/__([^_]+)__/g, '$1')

  // Convert italic
  text = text.replace(/\*([^*]+)\*/g, '$1')
  text = text.replace(/_([^_]+)_/g, '$1')

  // Convert code
  text = text.replace(/`([^`]+)`/g, '"$1"')

  // Convert links
  text = text.replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')

  // Convert images
  text = text.replace(/!\[([^\]]*)\]\([^)]+\)/g, '[Image: $1]')

  // Convert lists
  text = text.replace(/^\s*[-*+]\s+/gm, '• ')
  text = text.replace(/^\s*\d+\.\s+/gm, (match) => match.replace(/\d+/, (n) => n + '.'))

  // Convert blockquotes
  text = text.replace(/^>\s+/gm, '│ ')

  // Normalize whitespace
  text = text.replace(/\n{3,}/g, '\n\n')

  return text.trim()
}

/**
 * Check if text contains markdown formatting
 * @param text - Text to check
 * @returns True if text contains markdown
 */
export function containsMarkdown(text: string): boolean {
  const markdownPatterns = [
    /^#{1,6}\s+/m,           // Headings
    /\*\*[^*]+\*\*/,         // Bold
    /_[^_]+_/,               // Italic
    /`[^`]+`/,               // Code
    /\[[^\]]+\]\([^)]+\)/,   // Links
    /!\[[^\]]*\]\([^)]+\)/,  // Images
    /^\s*[-*+]\s+/m,         // Unordered lists
    /^\s*\d+\.\s+/m,         // Ordered lists
    /^>\s+/m,                // Blockquotes
    /```[\s\S]*?```/,        // Code blocks
  ]

  return markdownPatterns.some(pattern => pattern.test(text))
}

/**
 * Add syntax highlighting to code blocks
 * @param code - Code string
 * @param language - Language name
 * @returns HTML with syntax highlighting
 */
export function highlightCode(code: string, language: string): string {
  // This is a placeholder - in production you'd use a library like highlight.js or prism
  // For now, just wrap in pre/code tags with language class
  return `<pre><code class="language-${language}">${escapeHtml(code)}</code></pre>`
}

/**
 * Escape HTML special characters
 * @param html - HTML string
 * @returns Escaped HTML
 */
function escapeHtml(html: string): string {
  const div = document.createElement('div')
  div.textContent = html
  return div.innerHTML
}

/**
 * Create markdown link
 * @param text - Link text
 * @param url - URL
 * @param title - Optional title
 * @returns Markdown link
 */
export function createMarkdownLink(text: string, url: string, title?: string): string {
  return title ? `[${text}](${url} "${title}")` : `[${text}](${url})`
}

/**
 * Create markdown image
 * @param alt - Alt text
 * @param url - Image URL
 * @param title - Optional title
 * @returns Markdown image
 */
export function createMarkdownImage(alt: string, url: string, title?: string): string {
  return title ? `![${alt}](${url} "${title}")` : `![${alt}](${url})`
}

/**
 * Create markdown code block
 * @param code - Code string
 * @param language - Language name
 * @returns Markdown code block
 */
export function createMarkdownCodeBlock(code: string, language = ''): string {
  return `\`\`\`${language}\n${code}\n\`\`\``
}

/**
 * Create markdown list
 * @param items - List items
 * @param ordered - Whether list is ordered
 * @returns Markdown list
 */
export function createMarkdownList(items: string[], ordered = false): string {
  return items
    .map((item, _index) => {
      const prefix = ordered ? `${_index + 1}. ` : '- '
      return prefix + item
    })
    .join('\n')
}

export default {
  renderMarkdown,
  markdownToPlainText,
  extractCodeBlocks,
  extractInlineCode,
  extractLinks,
  extractImages,
  extractHeadings,
  generateTableOfContents,
  escapeMarkdown,
  unescapeMarkdown,
  truncateMarkdown,
  markdownToFormattedText,
  containsMarkdown,
  highlightCode,
  createMarkdownLink,
  createMarkdownImage,
  createMarkdownCodeBlock,
  createMarkdownList,
}

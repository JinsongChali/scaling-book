# Language Switch Feature Documentation

## Overview
This document describes the language switch feature that allows users to toggle between English and Chinese versions of the "How To Scale Your Model" book.

## Features

### 1. Language Toggle Button
- A language switch button has been added to the navigation bar
- The button displays "中文" when viewing English content and "English" when viewing Chinese content
- Located in the top navigation bar, next to the theme toggle button

### 2. Automatic Language Preference
- The site remembers the user's language preference using localStorage
- The preference persists across browser sessions
- Does not auto-redirect to avoid disrupting the user's browsing

### 3. Page Mapping
The following pages have been translated and mapped:

| English Page | Chinese Page |
|-------------|--------------|
| / (index) | /Chinese version/ |
| /roofline | /Chinese version/roofline |
| /tpus | /Chinese version/tpus |
| /sharding | /Chinese version/sharding |
| /transformers | /Chinese version/transformers |
| /training | /Chinese version/training |
| /applied-training | /Chinese version/applied-training |
| /inference | /Chinese version/inference |
| /applied-inference | /Chinese version/applied-inference |
| /profiling | /Chinese version/profiling |
| /jax-stuff | /Chinese version/jax-stuff |
| /conclusion | /Chinese version/conclusion |
| /gpus | /Chinese version/gpus |

## Implementation Details

### Files Modified
1. **`_includes/header.liquid`** - Added language switcher component to navigation
2. **`_includes/language-switcher.liquid`** - New component containing the language switch logic

### Files Created (Chinese Translations)
All Chinese translations are located in the `/Chinese version/` directory:
- index.md (介绍)
- roofline.md (屋顶线模型)
- tpus.md (如何理解TPU)
- sharding.md (分片矩阵及其乘法)
- transformers.md (如何理解Transformer)
- training.md (如何并行化Transformer进行训练)
- applied-training.md (在TPU上训练LLaMA 3)
- inference.md (关于Transformer推理的一切)
- applied-inference.md (在TPU上服务LLaMA 3-70B)
- profiling.md (如何对TPU程序进行性能分析)
- jax-stuff.md (用JAX编程TPU)
- conclusion.md (结论与延伸阅读)
- gpus.md (如何看待GPU)

## How It Works

### JavaScript Logic
The language switcher uses JavaScript to:
1. Store language preference in localStorage
2. Toggle between English and Chinese versions
3. Map current page to its translated counterpart
4. Handle URL redirection appropriately

### URL Structure
- English pages: `/scaling-book/[page-name]`
- Chinese pages: `/scaling-book/Chinese%20version/[page-name]`

## Usage

### For Users
1. Click the language button in the navigation bar
2. The page will automatically redirect to the corresponding language version
3. Your preference will be remembered for future visits

### For Developers
To add a new page translation:
1. Create the translated markdown file in `/Chinese version/`
2. Update the page mapping in `language-switcher.liquid`
3. Ensure proper navigation links in the translated file

## Technical Notes

### Browser Compatibility
- Uses localStorage (supported in all modern browsers)
- JavaScript required for functionality
- Graceful fallback if JavaScript is disabled (manual navigation still works)

### Dark Mode Support
The language button adapts to both light and dark themes automatically.

### Mobile Responsiveness
The language switcher is fully responsive and works on mobile devices.

## Future Enhancements

Potential improvements could include:
1. Auto-detection of browser language preference
2. URL-based language switching (e.g., `/en/` and `/zh/` prefixes)
3. Partial page translation support
4. Translation progress indicators

## Maintenance

### Adding New Translations
1. Translate the markdown file
2. Place in `/Chinese version/` directory
3. Update navigation links
4. Add mapping to language switcher

### Updating Existing Translations
Simply edit the corresponding file in `/Chinese version/`

## Contact
For questions or issues related to the translation or language switch feature, please refer to the main project documentation or contact the maintainers.
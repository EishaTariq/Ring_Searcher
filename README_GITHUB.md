# 💍 Ring Visual Search System

AI-powered visual search system for jewelry e-commerce websites. Find similar ring designs based on structural features with customizable gemstone options.

![Version](https://img.shields.io/badge/version-2.0-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

---

## ✨ Features

- 🔍 **Visual Search** - Upload any ring image and find similar designs
- 🎨 **Black & White Professional Theme** - Elegant minimalist interface
- ⚡ **Fast Performance** - Results in <1 second (5-10x faster than v1.0)
- 🔄 **Smart Deduplication** - Shows only unique designs (same ring different angles grouped)
- 💎 **Stone Customization** - Change gemstone types after finding base design
- 📊 **Match Percentages** - See similarity scores for each result
- 🎯 **Top 3 Results** - Focused, clean results display
- 📱 **Responsive Design** - Works on desktop, tablet, and mobile

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ring-visual-search.git
   cd ring-visual-search
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your catalog images**
   ```bash
   mkdir catalog
   # Copy your ring images to catalog/ folder
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open in browser**
   ```
   http://localhost:5000
   ```

---

## 📁 Project Structure

```
ring-visual-search/
├── ring_visual_search.py    # AI search engine
├── app.py                    # Flask web server
├── requirements.txt          # Python dependencies
├── templates/
│   └── index.html           # Web interface
├── catalog/                  # Your ring images (you create this)
├── uploads/                  # Temporary uploads (auto-created)
└── ring_features.pkl        # Features database (auto-created)
```

---

## 💻 Usage

### Web Interface

1. **Upload** - Click or drag & drop a ring image
2. **Wait** - System processes and searches (takes <1 second)
3. **View Results** - See top 3 unique matching designs with percentages
4. **Customize** - Click "Customize Stones" to change gemstone types

### Command Line (Alternative)

For faster testing without web interface:
```bash
python simple_ring_search.py
```

---

## 🎯 How It Works

### Image Processing Pipeline

1. **Preprocessing**
   - Noise removal (bilateral filtering)
   - Image resize to 256x256 (optimized for speed)
   - Pixel normalization

2. **Feature Extraction**
   - Color histograms (metal detection)
   - Edge detection (structure)
   - HOG features (shape patterns)
   - 44-dimensional feature vector

3. **Similarity Matching**
   - Cosine similarity comparison
   - 85% threshold for duplicate detection
   - Quality scoring (sharpness-based)

4. **Deduplication**
   - Groups same ring with different angles
   - Selects best quality image
   - Shows unique designs only

---

## ⚙️ Configuration

### Adjust Number of Results

Edit `app.py`, line ~78:
```python
top_k = 3  # Change to 5, 10, etc.
```

### Adjust Deduplication Threshold

Edit `app.py`, line ~80:
```python
threshold = 85.0  # Increase for less grouping (90-95)
                  # Decrease for more grouping (75-80)
```

### Change Image Size (Speed vs Accuracy)

Edit `ring_visual_search.py`, line ~89:
```python
img = cv2.resize(img, (256, 256))  # Smaller = faster
                                    # Larger = more accurate
```

---

## 📊 Performance

### Speed

| Catalog Size | Build Time | Search Time |
|--------------|------------|-------------|
| 50 images    | ~20 sec    | <1 sec      |
| 100 images   | ~45 sec    | <1 sec      |
| 200 images   | ~1.5 min   | <1 sec      |
| 500 images   | ~4 min     | 1-2 sec     |

### Accuracy

- **Exact matches**: 95-99% similarity
- **Very similar designs**: 85-94% similarity
- **Similar style**: 75-84% similarity
- **Different designs**: <75% similarity

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM       | 4GB     | 8GB         |
| CPU       | i3      | i5+         |
| Storage   | 2GB     | 10GB        |
| Catalog   | 10 imgs | 100-500 imgs|

---

## 🎨 Customization Options

### Stone Types Available

- Diamond (white)
- Sapphire (blue)
- Ruby (red)
- Emerald (green)
- Amethyst (purple)
- Topaz (various)
- Aquamarine (blue-green)
- Citrine (yellow)
- Garnet (red)
- Peridot (green)

### Stone Sizes

0.25ct, 0.5ct, 0.75ct, 1.0ct, 1.5ct, 2.0ct, 2.5ct, 3.0ct

### Positions

Center, Side, Halo, Band, Accent, Custom

---

## 🐛 Troubleshooting

### "No images found in catalog"

**Solution**: Add images to `catalog/` folder
```bash
mkdir catalog
cp /path/to/your/rings/* catalog/
```

### "Module not found" errors

**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

### Slow performance

**Solutions**:
- Reduce catalog size (under 500 images)
- Use smaller images (under 2MB each)
- Increase threshold to 90% (less deduplication checks)

### Port 5000 already in use

**Solution**: Change port in `app.py`
```python
app.run(debug=True, port=5001)  # Use different port
```

---

## 📚 Documentation

- **DATASET_SETUP_GUIDE.md** - Complete setup instructions (Urdu + English)
- **SPEED_OPTIMIZATION_GUIDE.md** - Performance optimization details
- **ACCURACY_GUIDE.md** - How accuracy is calculated
- **VISUAL_WALKTHROUGH.md** - Screenshots and UI guide

---

## 🔧 Tech Stack

- **Backend**: Python 3.8+, Flask
- **Computer Vision**: OpenCV, scikit-image
- **Machine Learning**: scikit-learn (cosine similarity)
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Image Processing**: Pillow, NumPy

---

## 📈 Version History

### v2.0 (Current - Optimized)
- 5-10x faster performance
- Reduced image size (512→256)
- Simplified features (130→44 dims)
- Removed SSIM checks
- Clean, minimal output

### v1.0
- Initial release
- Full feature set
- SSIM-based deduplication
- Comprehensive quality scoring

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- Built with Python and Flask
- Uses OpenCV for image processing
- Inspired by modern e-commerce visual search systems

---

## 📞 Support

For issues or questions:
- Open an issue on GitHub
- Check documentation in `/docs` folder
- Review troubleshooting section above

---

## 🌟 Star This Repo

If you find this project useful, please consider giving it a star! ⭐

---

**Made with ❤️ for jewelry e-commerce**

**بہت شکریہ! Thank you!**

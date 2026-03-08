# ✅ Setup Checklist - SmartDoc AI

Use this checklist to ensure everything is set up correctly.

## 📋 Pre-Installation Checklist

- [ ] **Python 3.8+** installed
  ```bash
  python --version  # Should show 3.8 or higher
  ```

- [ ] **pip** package manager available
  ```bash
  pip --version
  ```

- [ ] **Ollama** installed
  ```bash
  ollama --version
  ```

---

## 🔧 Installation Checklist

- [ ] **Step 1:** Navigate to project directory
  ```bash
  cd /home/thinh/Python/ChatBotAI
  ```

- [ ] **Step 2:** Create virtual environment (recommended)
  ```bash
  python -m venv venv
  source venv/bin/activate  # Linux/Mac
  # OR
  venv\Scripts\activate  # Windows
  ```

- [ ] **Step 3:** Install dependencies
  ```bash
  pip install -r requirements.txt
  ```

- [ ] **Step 4:** Verify installations
  ```bash
  pip list | grep streamlit
  pip list | grep langchain
  pip list | grep faiss
  ```

---

## 🤖 Ollama Setup Checklist

- [ ] **Step 1:** Start Ollama server
  ```bash
  ollama serve
  # Keep this terminal open
  ```

- [ ] **Step 2:** Pull qwen2.5:7b model (in new terminal)
  ```bash
  ollama pull qwen2.5:7b
  ```

- [ ] **Step 3:** Verify model is available
  ```bash
  ollama list
  # Should show qwen2.5:7b in the list
  ```

- [ ] **Step 4:** Test Ollama connection
  ```bash
  curl http://localhost:11434/api/tags
  # Should return JSON with model info
  ```

---

## 🚀 First Run Checklist

- [ ] **Step 1:** Start the application
  ```bash
  streamlit run app.py
  ```

- [ ] **Step 2:** Browser opens automatically at `http://localhost:8501`
  - If not, manually open the URL

- [ ] **Step 3:** Check initial state
  - [ ] Sidebar shows "🟡 No Documents"
  - [ ] Three tabs visible: Chat, Documents, Settings
  - [ ] No errors in terminal

- [ ] **Step 4:** Test Ollama connection (in Settings tab)
  - [ ] Click "🔌 Test Ollama Connection"
  - [ ] Should show "Connected successfully"

---

## 📄 Upload Test Checklist

- [ ] **Step 1:** Go to "📄 Documents" tab

- [ ] **Step 2:** Prepare a test file
  - [ ] Have a PDF, DOCX, or TXT file ready
  - [ ] File size < 10 MB

- [ ] **Step 3:** Upload document
  - [ ] Click "Choose a document"
  - [ ] Select your file
  - [ ] Click "📤 Process Document"

- [ ] **Step 4:** Verify processing
  - [ ] See "Processing document..." spinner
  - [ ] Success message appears
  - [ ] Balloons animation plays 🎉
  - [ ] Sidebar shows "🟢 Documents Loaded"

---

## 💬 Chat Test Checklist

- [ ] **Step 1:** Go to "💬 Chat" tab

- [ ] **Step 2:** Verify interface
  - [ ] Chat input visible at bottom
  - [ ] No "upload documents first" message
  - [ ] Empty chat history message shown

- [ ] **Step 3:** Ask first question
  - [ ] Type a simple question about your document
  - [ ] Press Enter or click send
  - [ ] See "🤔 Thinking..." spinner

- [ ] **Step 4:** Verify response
  - [ ] Answer appears in chat
  - [ ] "📚 View Sources" expandable section shown
  - [ ] Sources display document excerpts
  - [ ] Message counter in sidebar increases

- [ ] **Step 5:** Test conversation
  - [ ] Ask follow-up question
  - [ ] Both Q&A pairs visible in history

---

## ⚙️ Settings Test Checklist

- [ ] **Step 1:** Go to "⚙️ Settings" tab

- [ ] **Step 2:** View chunk settings
  - [ ] Chunk Size slider visible (500-2000)
  - [ ] Chunk Overlap slider visible (50-300)

- [ ] **Step 3:** Adjust settings
  - [ ] Move Chunk Size to 1500
  - [ ] Move Chunk Overlap to 150
  - [ ] Click "💾 Apply Chunk Settings"
  - [ ] See success message

- [ ] **Step 4:** Verify system info
  - [ ] Embedding model name shown
  - [ ] Vector database type shown
  - [ ] Ollama URL displayed

---

## 🔍 Troubleshooting Checklist

### ❌ "Cannot connect to Ollama"

- [ ] Check Ollama is running
  ```bash
  ps aux | grep ollama  # Linux/Mac
  # OR
  tasklist | findstr ollama  # Windows
  ```

- [ ] Restart Ollama
  ```bash
  pkill ollama  # Stop
  ollama serve  # Start
  ```

### ❌ "Import Error: No module named..."

- [ ] Reinstall requirements
  ```bash
  pip install -r requirements.txt --force-reinstall
  ```

### ❌ "Vector store not initialized"

- [ ] Upload a document first in Documents tab
- [ ] Check for upload errors in terminal

### ❌ "File too large"

- [ ] Check file size
- [ ] Split large documents into smaller parts
- [ ] Or increase MAX_FILE_SIZE_MB in constants.py

---

## ✅ Final Verification

- [ ] **Application starts** without errors
- [ ] **Ollama connection** works
- [ ] **Document upload** succeeds
- [ ] **Questions get answered** correctly
- [ ] **Sources displayed** properly
- [ ] **Settings adjustable** without errors
- [ ] **Chat history** persists during session
- [ ] **Clear functions** work correctly

---

## 🎉 Success Indicators

If you see all these, you're good to go:

✅ Sidebar status: **🟢 Documents Loaded**
✅ Ollama test: **Connected successfully**
✅ Upload test: **Balloons animation** 🎉
✅ Chat test: **Answer received** with sources
✅ No error messages in terminal
✅ All features working smoothly

---

## 📞 Getting Help

If you're stuck:

1. **Check terminal output** for error messages
2. **Review QUICKSTART.md** for detailed instructions
3. **Check ARCHITECTURE.md** to understand the flow
4. **Read AGENTS.md** for design principles

---

## 🎯 Next Steps

Once everything is working:

- [ ] Try different document types (PDF, DOCX, TXT)
- [ ] Experiment with chunk settings
- [ ] Test with longer documents
- [ ] Ask complex multi-part questions
- [ ] Test Vietnamese and English queries
- [ ] Review source citations for accuracy

---

**Happy coding! 🚀**

*Last updated: 2026-03-07*

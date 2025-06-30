#!/usr/bin/env ruby
# frozen_string_literal: true

# ===============================================================================
# Contextual Chunk Headers RAG (Phiên bản Ruby)
# ===============================================================================
#
# Contextual Chunk Headers (CCH) giải quyết vấn đề thiếu thông tin ngữ cảnh
# tổng thể của các chunks bằng cách tự động tạo ra các header mô tả cho
# từng chunk. Các header này giúp LLM hiểu rõ hơn về nội dung và bối cảnh
# của từng chunk.
#
# Quy trình:
# 1. Trích xuất văn bản từ PDF.
# 2. Chia thành các chunk nhỏ.
# 3. Tạo header cho mỗi chunk bằng LLM.
# 4. Kết hợp header với nội dung chunk.
# 5. Tạo embeddings cho chunk đã có header.
# 6. Thực hiện RAG với enhanced chunks.

require 'net/http'
require 'uri'
require 'json'
require 'matrix'
require 'pdf-reader'

# ===============================================================================
# Thiết lập Môi trường
# ===============================================================================
puts "=== Contextual Chunk Headers RAG ==="

# Cấu hình
OPENAI_API_KEY = ENV['OPENAI_API_KEY']
BASE_URL = 'https://api.studio.nebius.com/v1/'
EMBEDDING_MODEL = 'BAAI/bge-en-icl'
CHAT_MODEL = 'meta-llama/Llama-3.3-70B-Instruct'

# Kiểm tra API key
if OPENAI_API_KEY.nil? || OPENAI_API_KEY.empty?
  puts "Cảnh báo: Biến môi trường OPENAI_API_KEY chưa được đặt"
  exit 1
end

# ===============================================================================
# Hàm Trích xuất và Chia nhỏ
# ===============================================================================

def extract_text_from_pdf(pdf_path)
  # Trích xuất văn bản từ file PDF.
  #
  # Args:
  #   pdf_path (String): Đường dẫn đến file PDF.
  #
  # Returns:
  #   String: Văn bản được trích xuất từ PDF.
  reader = PDF::Reader.new(pdf_path)
  all_text = ""

  reader.pages.each do |page|
    all_text += page.text + " "
  end

  all_text.strip
end

def chunk_text(text, chunk_size = 1000, overlap = 200)
  # Chia văn bản thành các chunk có overlap.
  #
  # Args:
  #   text (String): Văn bản cần chia nhỏ.
  #   chunk_size (Integer): Số ký tự trong mỗi chunk.
  #   overlap (Integer): Số ký tự overlap giữa các chunk.
  #
  # Returns:
  #   Array<String>: Mảng các chunk văn bản.
  chunks = []
  step_size = chunk_size - overlap

  (0...text.length).step(step_size) do |i|
    chunk = text[i, chunk_size]
    chunks << chunk unless chunk.empty?
  end

  chunks
end

# ===============================================================================
# Hàm API Client
# ===============================================================================

def make_api_request(endpoint, payload)
  # Thực hiện API request đến OpenAI-compatible endpoint.
  uri = URI("#{BASE_URL}#{endpoint}")
  http = Net::HTTP.new(uri.host, uri.port)
  http.use_ssl = true

  request = Net::HTTP::Post.new(uri)
  request['Authorization'] = "Bearer #{OPENAI_API_KEY}"
  request['Content-Type'] = 'application/json'
  request.body = payload.to_json

  response = http.request(request)

  if response.code.to_i == 200
    JSON.parse(response.body)
  else
    puts "Lỗi API: #{response.code} - #{response.body}"
    nil
  end
end

def create_embeddings(text_list, model = EMBEDDING_MODEL)
  # Tạo embeddings cho danh sách văn bản.
  #
  # Args:
  #   text_list (Array): Mảng văn bản đầu vào.
  #   model (String): Mô hình embedding.
  #
  # Returns:
  #   Array: Mảng embeddings.
  payload = {
    model: model,
    input: text_list
  }

  response = make_api_request('embeddings', payload)
  return [] unless response && response['data']

  response['data'].map { |item| item['embedding'] }
end

def generate_response(system_prompt, user_message, model = CHAT_MODEL)
  # Tạo phản hồi từ mô hình AI.
  payload = {
    model: model,
    temperature: 0,
    messages: [
      { role: 'system', content: system_prompt },
      { role: 'user', content: user_message }
    ]
  }

  response = make_api_request('chat/completions', payload)
  return nil unless response && response['choices'] && !response['choices'].empty?

  response['choices'][0]['message']['content']
end

# ===============================================================================
# Kỹ thuật chính: Tạo Contextual Headers
# ===============================================================================

def generate_chunk_header(chunk, model = CHAT_MODEL)
  # Tạo header mô tả cho một chunk văn bản bằng LLM.
  #
  # Đây là kỹ thuật cốt lõi của bài này - sử dụng LLM để tự động
  # tạo ra một tiêu đề ngắn gọn và mô tả chính xác nội dung của chunk.
  #
  # Args:
  #   chunk (String): Nội dung chunk cần tạo header.
  #   model (String): Mô hình LLM để sử dụng.
  #
  # Returns:
  #   String: Header được tạo cho chunk, hoặc nil nếu thất bại.
  system_prompt = "
  Bạn là một chuyên gia trong việc tạo tiêu đề và tóm tắt văn bản.
  Nhiệm vụ của bạn là tạo ra một header (tiêu đề) ngắn gọn và mô tả
  chính xác nội dung của đoạn văn bản được cung cấp.

  HƯỚNG DẪN:
  - Header phải ngắn gọn (tối đa 10-15 từ)
  - Phải mô tả chính xác chủ đề/nội dung chính của đoạn văn
  - Sử dụng ngôn ngữ rõ ràng, dễ hiểu
  - Tránh từ ngữ chung chung hoặc mơ hồ
  - CHỈ trả về header, không giải thích thêm

  Ví dụ header tốt:
  - \"AI Applications in Medical Diagnosis\"
  - \"Machine Learning vs Traditional Programming Methods\"
  - \"Ethical Concerns in Artificial Intelligence Development\"
  "

  user_message = "
  Hãy tạo một header ngắn gọn và mô tả chính xác cho đoạn văn bản sau:

  #{chunk}

  Header:
  "

  header = generate_response(system_prompt, user_message, model)

  if header
    # Clean up header - remove quotes, extra whitespace, "Header:" prefix
    cleaned_header = header.strip
                          .gsub(/^["']|["']$/, '')  # Remove quotes at start/end
                          .gsub(/^Header:\s*/i, '') # Remove "Header:" prefix
                          .strip

    puts "Generated header: #{cleaned_header}"
    cleaned_header
  else
    puts "Failed to generate header for chunk"
    nil
  end
end

def create_enhanced_chunks(text_chunks)
  # Tạo enhanced chunks bằng cách thêm contextual headers.
  #
  # Quy trình:
  # 1. Lặp qua từng chunk gốc
  # 2. Tạo header cho chunk đó bằng LLM
  # 3. Kết hợp header với nội dung chunk
  # 4. Trả về danh sách enhanced chunks
  #
  # Args:
  #   text_chunks (Array): Mảng các chunk văn bản gốc.
  #
  # Returns:
  #   Hash: {
  #     enhanced_chunks: Array - Chunks đã có header,
  #     headers: Array - Danh sách các headers được tạo,
  #     original_chunks: Array - Chunks gốc
  #   }
  puts "\n--- Bắt đầu tạo Contextual Headers ---"

  enhanced_chunks = []
  headers = []
  total_chunks = text_chunks.length

  text_chunks.each_with_index do |chunk, index|
    puts "\nXử lý chunk #{index + 1}/#{total_chunks}..."

    # Tạo header cho chunk
    header = generate_chunk_header(chunk)

    if header
      # Tạo enhanced chunk với header
      enhanced_chunk = "
## #{header}

#{chunk}
"
      enhanced_chunks << enhanced_chunk
      headers << header

      puts "✓ Chunk #{index + 1} completed with header: #{header[0..50]}..."
    else
      # Fallback: sử dụng chunk gốc nếu không tạo được header
      puts "⚠ Failed to generate header for chunk #{index + 1}, using original"
      enhanced_chunks << chunk
      headers << "Content Section #{index + 1}"
    end

    # Delay nhỏ để tránh rate limiting
    sleep(0.5)
  end

  puts "\n--- Hoàn thành tạo Headers ---"
  puts "Tổng số headers tạo thành công: #{headers.count { |h| !h.start_with?('Content Section') }}"

  {
    enhanced_chunks: enhanced_chunks,
    headers: headers,
    original_chunks: text_chunks
  }
end

# ===============================================================================
# Hàm Tìm kiếm Semantic
# ===============================================================================

def cosine_similarity(vec1, vec2)
  # Tính cosine similarity giữa hai vector.
  v1 = Vector[*vec1]
  v2 = Vector[*vec2]

  dot_product = v1.inner_product(v2)
  magnitude1 = Math.sqrt(v1.inner_product(v1))
  magnitude2 = Math.sqrt(v2.inner_product(v2))

  return 0.0 if magnitude1 == 0.0 || magnitude2 == 0.0

  dot_product / (magnitude1 * magnitude2)
end

def semantic_search(query, enhanced_chunks, embeddings, k = 5)
  # Tìm kiếm semantic trên enhanced chunks (có headers).
  #
  # Args:
  #   query (String): Câu hỏi của người dùng.
  #   enhanced_chunks (Array): Chunks đã có headers.
  #   embeddings (Array): Embeddings của enhanced chunks.
  #   k (Integer): Số chunks cần trả về.
  #
  # Returns:
  #   Array: Top-k enhanced chunks liên quan nhất.
  # Tạo embedding cho query
  query_embeddings = create_embeddings([query])
  return [] if query_embeddings.empty?

  query_embedding = query_embeddings[0]

  # Tính similarity scores
  similarities = embeddings.each_with_index.map do |chunk_emb, i|
    similarity = cosine_similarity(query_embedding, chunk_emb)
    [i, similarity, enhanced_chunks[i]]
  end

  # Sắp xếp và lấy top k
  top_results = similarities.sort_by { |_, score, _| -score }.first(k)

  puts "\nTop #{k} kết quả tìm kiếm:"
  top_results.each_with_index do |(index, score, chunk), i|
    header = chunk.split("\n").find { |line| line.start_with?('## ') }&.gsub('## ', '') || 'No header'
    puts "#{i + 1}. Score: #{score.round(4)} - #{header}"
  end

  top_results.map { |_, _, chunk| chunk }
end

# ===============================================================================
# So sánh với Basic RAG
# ===============================================================================

def compare_basic_vs_contextual_headers(query, original_chunks, enhanced_chunks,
                                        original_embeddings, enhanced_embeddings)
  # So sánh kết quả giữa Basic RAG và Contextual Headers RAG.
  puts "\n=== So sánh Basic RAG vs Contextual Headers RAG ==="
  puts "Query: #{query}"
  puts "=" * 80

  # 1. Basic RAG (không có headers)
  puts "\n1. BASIC RAG (KHÔNG CÓ HEADERS):"
  puts "-" * 50
  basic_results = semantic_search(query, original_chunks, original_embeddings, 3)

  if basic_results.any?
    puts "Top chunk (Basic RAG):"
    puts basic_results[0][0..300] + "..."
    puts "Độ dài: #{basic_results[0].length} ký tự"
  end

  # 2. Contextual Headers RAG (có headers)
  puts "\n2. CONTEXTUAL HEADERS RAG (CÓ HEADERS):"
  puts "-" * 50
  contextual_results = semantic_search(query, enhanced_chunks, enhanced_embeddings, 3)

  if contextual_results.any?
    puts "Top chunk (Contextual Headers RAG):"
    puts contextual_results[0][0..400] + "..."
    puts "Độ dài: #{contextual_results[0].length} ký tự"

    # Hiển thị header được tạo
    header_line = contextual_results[0].split("\n").find { |line| line.start_with?('## ') }
    if header_line
      puts "Header: #{header_line.gsub('## ', '')}"
    end
  end

  puts "=" * 80

  { basic: basic_results, contextual: contextual_results }
end

# ===============================================================================
# Tạo phản hồi với Enhanced Chunks
# ===============================================================================

def generate_contextual_response(query, enhanced_chunks)
  # Tạo phản hồi AI sử dụng enhanced chunks có headers.
  system_prompt = "
  Bạn là một AI assistant thông minh với khả năng phân tích văn bản có cấu trúc.
  Bạn sẽ nhận được các đoạn văn bản có kèm headers mô tả nội dung.

  CHÚ Ý:
  - Headers (bắt đầu bằng ##) cung cấp thông tin tóm tắt về nội dung
  - Sử dụng cả headers và nội dung để hiểu đầy đủ ngữ cảnh
  - Ưu tiên thông tin từ chunks có headers liên quan nhất đến câu hỏi
  - Nếu thông tin không đủ, hãy nói rõ những gì còn thiếu
  - Đưa ra câu trả lời dựa trên bằng chứng từ văn bản
  "

  # Chuẩn bị context từ enhanced chunks
  context = enhanced_chunks.each_with_index.map do |chunk, i|
    "=== CONTEXT #{i + 1} ===\n#{chunk}"
  end.join("\n\n")

  user_message = "#{context}\n\n=== CÂU HỎI ===\n#{query}"

  generate_response(system_prompt, user_message)
end

# ===============================================================================
# Demo chính
# ===============================================================================

def run_contextual_headers_demo
  # Chạy demo hoàn chỉnh Contextual Chunk Headers RAG.
  puts "\n=== Demo Contextual Chunk Headers RAG ==="

  # Bước 1: Trích xuất văn bản từ PDF
  puts "\n1. Trích xuất văn bản từ PDF..."
  pdf_path = "data/AI_Information.pdf"
  extracted_text = extract_text_from_pdf(pdf_path)
  puts "Độ dài văn bản: #{extracted_text.length} ký tự"

  # Bước 2: Chia thành chunks
  puts "\n2. Chia văn bản thành chunks..."
  original_chunks = chunk_text(extracted_text, 1000, 200)
  puts "Số chunks được tạo: #{original_chunks.length}"

  # Giới hạn số chunks để demo (tránh quá nhiều API calls)
  demo_chunks = original_chunks.first(5)
  puts "Demo với #{demo_chunks.length} chunks đầu tiên"

  # Bước 3: Tạo contextual headers
  puts "\n3. Tạo contextual headers cho các chunks..."
  enhanced_data = create_enhanced_chunks(demo_chunks)
  enhanced_chunks = enhanced_data[:enhanced_chunks]
  headers = enhanced_data[:headers]

  puts "\nHeaders được tạo:"
  headers.each_with_index do |header, i|
    puts "#{i + 1}. #{header}"
  end

  # Bước 4: Tạo embeddings cho cả original và enhanced chunks
  puts "\n4. Tạo embeddings..."
  puts "Tạo embeddings cho original chunks..."
  original_embeddings = create_embeddings(demo_chunks)

  puts "Tạo embeddings cho enhanced chunks..."
  enhanced_embeddings = create_embeddings(enhanced_chunks)

  return unless original_embeddings.any? && enhanced_embeddings.any?

  # Bước 5: So sánh với nhiều queries
  queries = [
    "What are the main applications of artificial intelligence?",
    "How does machine learning work in practice?",
    "What are the benefits and risks of AI technology?"
  ]

  queries.each_with_index do |query, i|
    puts "\n" + "=" * 100
    puts "DEMO #{i + 1}: #{query}"
    puts "=" * 100

    # So sánh Basic vs Contextual Headers
    comparison_results = compare_basic_vs_contextual_headers(
      query,
      demo_chunks,
      enhanced_chunks,
      original_embeddings,
      enhanced_embeddings
    )

    # Tạo phản hồi với Contextual Headers
    if comparison_results[:contextual].any?
      puts "\n3. PHẢN HỒI VỚI CONTEXTUAL HEADERS:"
      puts "-" * 60
      contextual_response = generate_contextual_response(query, comparison_results[:contextual])

      if contextual_response
        puts contextual_response
      else
        puts "Không thể tạo phản hồi"
      end
    end

    puts "\n" + "=" * 100
  end
end

# ===============================================================================
# Demo phân tích Headers
# ===============================================================================

def analyze_generated_headers
  # Demo và phân tích chất lượng của các headers được tạo.
  puts "\n=== Phân tích Headers được tạo ==="

  # Load một số chunks mẫu để phân tích
  pdf_path = "data/AI_Information.pdf"
  extracted_text = extract_text_from_pdf(pdf_path)
  sample_chunks = chunk_text(extracted_text, 800, 150).first(3)

  puts "Phân tích #{sample_chunks.length} chunks mẫu:\n"

  sample_chunks.each_with_index do |chunk, i|
    puts "=" * 60
    puts "CHUNK #{i + 1}:"
    puts "=" * 60

    # Hiển thị nội dung chunk (200 ký tự đầu)
    puts "Nội dung gốc:"
    puts chunk[0..200] + (chunk.length > 200 ? "..." : "")
    puts

    # Tạo header
    puts "Đang tạo header..."
    header = generate_chunk_header(chunk)

    if header
      puts "Header được tạo: \"#{header}\""

      # Phân tích header
      puts "\nPhân tích header:"
      puts "- Độ dài: #{header.length} ký tự"
      puts "- Số từ: #{header.split.length} từ"
      puts "- Có mô tả rõ ràng: #{header.include?('AI') || header.include?('Machine') || header.include?('Technology') ? 'Có' : 'Cần kiểm tra'}"
    else
      puts "❌ Không thể tạo header"
    end

    puts "\n"
  end
end

# ===============================================================================
# Main Execution
# ===============================================================================

if __FILE__ == $0
  puts "Contextual Chunk Headers RAG bằng Ruby"
  puts "=" * 60

  begin
    # Demo chính
    run_contextual_headers_demo

    # Phân tích headers
    puts "\n\n"
    analyze_generated_headers

    puts "\n=== HOÀN THÀNH DEMO CONTEXTUAL CHUNK HEADERS RAG ==="
  rescue => e
    puts "Lỗi trong quá trình thực thi: #{e.message}"
    puts e.backtrace
  end
end

#!/usr/bin/env ruby
# frozen_string_literal: true

# ===============================================================================
# Giới thiệu về Simple RAG (Phiên bản Ruby)
# ===============================================================================
#
# Retrieval-Augmented Generation (RAG) là một phương pháp kết hợp giữa
# truy xuất thông tin và mô hình sinh văn bản. Nó cải thiện hiệu suất của
# các mô hình ngôn ngữ bằng cách kết hợp kiến thức bên ngoài, giúp tăng
# độ chính xác và tính factual.
#
# Trong thiết lập Simple RAG, chúng ta thực hiện các bước sau:
#
# 1. **Thu thập dữ liệu**: Tải và xử lý dữ liệu văn bản.
# 2. **Chia nhỏ**: Chia dữ liệu thành các đoạn nhỏ hơn để cải thiện hiệu suất truy xuất.
# 3. **Tạo Embedding**: Chuyển đổi các đoạn văn bản thành biểu diễn số bằng mô hình embedding.
# 4. **Tìm kiếm Semantic**: Truy xuất các đoạn liên quan dựa trên câu hỏi của người dùng.
# 5. **Tạo phản hồi**: Sử dụng mô hình ngôn ngữ để tạo phản hồi dựa trên văn bản đã truy xuất.
#
# Script Ruby này triển khai phương pháp Simple RAG, đánh giá phản hồi của mô hình,
# và khám phá các cải tiến khác nhau.

require 'net/http'
require 'uri'
require 'json'
require 'matrix'
require 'pdf-reader'

# ===============================================================================
# Thiết lập Môi trường
# ===============================================================================
puts "=== Thiết lập Môi trường Simple RAG ==="

# Đọc file .env nếu tồn tại
if File.exist?('.env')
  File.readlines('.env').each do |line|
    key, value = line.strip.split('=', 2)
    ENV[key] = value if key && value
  end
end

# Cấu hình
OPENAI_API_KEY = ENV['OPENAI_API_KEY']
BASE_URL = 'https://api.studio.nebius.com/v1/'
EMBEDDING_MODEL = 'BAAI/bge-en-icl'
CHAT_MODEL = 'meta-llama/Llama-3.3-70B-Instruct'

# Kiểm tra xem API key có sẵn không
if OPENAI_API_KEY.nil? || OPENAI_API_KEY.empty?
  puts "Cảnh báo: Biến môi trường OPENAI_API_KEY chưa được đặt"
else
  puts "API key đã được tải thành công"
end

# ===============================================================================
# Trích xuất Văn bản từ File PDF
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

# ===============================================================================
# Chia nhỏ Văn bản đã Trích xuất
# ===============================================================================

def chunk_text(text, chunk_size, overlap)
  # Chia văn bản thành các đoạn có kích thước chunk_size ký tự với overlap.
  #
  # Args:
  #   text (String): Văn bản cần chia nhỏ.
  #   chunk_size (Integer): Số ký tự trong mỗi đoạn.
  #   overlap (Integer): Số ký tự overlap giữa các đoạn.
  #
  # Returns:
  #   Array<String>: Mảng các đoạn văn bản.

  chunks = []
  step_size = chunk_size - overlap

  (0...text.length).step(step_size) do |i|
    chunk = text[i, chunk_size]
    chunks << chunk unless chunk.empty?
  end

  chunks
end

# ===============================================================================
# Các Hàm OpenAI API Client
# ===============================================================================

def make_api_request(endpoint, payload)
  """
  Thực hiện API request đến OpenAI-compatible endpoint.

  Args:
    endpoint (String): Đường dẫn API endpoint
    payload (Hash): Dữ liệu request

  Returns:
    Hash: JSON response đã được parse
  """

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

def create_embeddings(text, model = EMBEDDING_MODEL)
  """
  Tạo embeddings cho văn bản sử dụng mô hình được chỉ định.

  Args:
    text (String or Array): Văn bản đầu vào cần tạo embeddings.
    model (String): Mô hình được sử dụng để tạo embeddings.

  Returns:
    Hash: Response từ API chứa embeddings.
  """

  payload = {
    model: model,
    input: text
  }

  make_api_request('embeddings', payload)
end

def generate_response(system_prompt, user_message, model = CHAT_MODEL)
  """
  Tạo phản hồi từ mô hình AI dựa trên system prompt và user message.

  Args:
    system_prompt (String): System prompt để hướng dẫn hành vi của AI.
    user_message (String): Tin nhắn hoặc câu hỏi của người dùng.
    model (String): Mô hình được sử dụng để tạo phản hồi.

  Returns:
    Hash: Phản hồi từ mô hình AI.
  """

  payload = {
    model: model,
    temperature: 0,
    messages: [
      { role: 'system', content: system_prompt },
      { role: 'user', content: user_message }
    ]
  }

  make_api_request('chat/completions', payload)
end

# ===============================================================================
# Các Hàm Tìm kiếm Semantic
# ===============================================================================
# Giải thích chi tiết về Cosine Similarity trong RAG
# =================================================
#
# Cosine similarity là phương pháp đo lường độ tương tự giữa hai vector bằng cách tính cosine của góc giữa chúng.
# Đây là một trong những phương pháp phổ biến nhất trong tìm kiếm semantic và RAG systems.
#
# Công thức: cos(θ) = (A · B) / (||A|| × ||B||)
#
# Trong đó:
# - A · B: Dot product (tích vô hướng) của hai vector
# - ||A||, ||B||: Magnitude (độ dài Euclidean) của mỗi vector
# - θ: Góc giữa hai vector trong không gian nhiều chiều
#
# Kết quả và ý nghĩa:
# - 1.0: Hai vector hoàn toàn giống nhau (góc 0°) - văn bản có nghĩa giống hệt nhau
# - 0.0: Hai vector vuông góc (góc 90°) - văn bản không liên quan
# - -1.0: Hai vector ngược hướng (góc 180°) - văn bản có nghĩa đối lập
#
# Ưu điểm trong RAG:
# - Không bị ảnh hưởng bởi độ dài văn bản (chỉ quan tâm đến hướng semantic)
# - Chuẩn hóa tự động (kết quả luôn trong khoảng [-1, 1])
# - Hiệu quả tính toán và phù hợp với high-dimensional embeddings
# - Phản ánh tốt mối quan hệ semantic giữa các đoạn văn bản
#
# Ví dụ thực tế:
# - Query: "What is machine learning?"
# - Chunk 1: "Machine learning is a subset of AI..." → similarity: 0.85
# - Chunk 2: "The weather today is sunny..." → similarity: 0.12
# =================================================

def cosine_similarity(vec1, vec2)
  # Tính cosine similarity giữa hai vector.
  #
  # Args:
  #   vec1 (Array): Vector thứ nhất.
  #   vec2 (Array): Vector thứ hai.
  #
  # Returns:
  #   Float: Cosine similarity giữa hai vector.

  # Chuyển đổi thành vector nếu cần
  v1 = vec1.is_a?(Array) ? Vector[*vec1] : vec1
  v2 = vec2.is_a?(Array) ? Vector[*vec2] : vec2

  # Tính dot product và magnitudes
  dot_product = v1.inner_product(v2)
  magnitude1 = Math.sqrt(v1.inner_product(v1))
  magnitude2 = Math.sqrt(v2.inner_product(v2))

  # Trả về cosine similarity
  dot_product / (magnitude1 * magnitude2)
end

def semantic_search(query, text_chunks, embeddings, k = 5)

  # Thực hiện tìm kiếm semantic trên các đoạn văn bản sử dụng query và embeddings.
  #
  # Args:
  #   query (String): Câu hỏi để tìm kiếm semantic.
  #   text_chunks (Array<String>): Mảng các đoạn văn bản để tìm kiếm.
  #   embeddings (Array): Mảng embeddings cho các đoạn văn bản.
  #   k (Integer): Số lượng đoạn văn bản liên quan nhất cần trả về.
  #
  # Returns:
  #   Array<String>: Mảng top k đoạn văn bản liên quan nhất dựa trên query.

  # Tạo embedding cho query
  query_response = create_embeddings(query)
  return [] unless query_response && query_response['data']

  query_embedding = query_response['data'][0]['embedding']
  similarity_scores = []

  # Tính điểm similarity giữa query embedding và mỗi text chunk embedding
  embeddings.each_with_index do |chunk_embedding, i|
    similarity_score = cosine_similarity(query_embedding, chunk_embedding['embedding'])
    similarity_scores << [i, similarity_score]
  end

  # Sắp xếp điểm similarity theo thứ tự giảm dần và lấy top k
  top_indices = similarity_scores.sort_by { |_, score| -score }.first(k).map(&:first)

  # Trả về top k đoạn văn bản liên quan nhất
  top_indices.map { |index| text_chunks[index] }
end

# ===============================================================================
# Triển khai RAG Chính
# ===============================================================================

def run_simple_rag_demo
  # Chạy demo hoàn chỉnh của hệ thống Simple RAG.

  puts "\n=== Demo Simple RAG ==="

  # Bước 1: Trích xuất văn bản từ PDF
  puts "\n1. Trích xuất văn bản từ PDF..."
  pdf_path = "data/AI_Information.pdf"
  extracted_text = extract_text_from_pdf(pdf_path)
  puts "Độ dài văn bản đã trích xuất: #{extracted_text.length} ký tự"

  # Bước 2: Chia nhỏ văn bản
  puts "\n2. Chia nhỏ văn bản..."
  text_chunks = chunk_text(extracted_text, 1000, 200)
  puts "Số lượng đoạn văn bản: #{text_chunks.length}"
  puts "\nĐoạn văn bản đầu tiên:"
  puts text_chunks[0][0..500] + "..." if text_chunks[0]

  # Bước 3: Tạo embeddings
  puts "\n3. Tạo embeddings cho các đoạn văn bản..."
  embeddings_response = create_embeddings(text_chunks)

  unless embeddings_response && embeddings_response['data']
    puts "Không thể tạo embeddings. Vui lòng kiểm tra API key và kết nối."
    return
  end

  embeddings = embeddings_response['data']
  puts "Đã tạo #{embeddings.length} embeddings"

    # Bước 4: Tải dữ liệu validation và thực hiện tìm kiếm
  puts "\n4. Tải dữ liệu validation và thực hiện tìm kiếm semantic..."

  # Câu hỏi mẫu để demo
  sample_queries = [
    "What is 'Explainable AI' and why is it considered important?",
    "What are the main challenges in AI development?",
    "How has AI research evolved over time?"
  ]

  query = sample_queries[0]
  puts "Câu hỏi: #{query}"

  # Thực hiện tìm kiếm semantic
  top_chunks = semantic_search(query, text_chunks, embeddings, 2)

  puts "\nCác đoạn liên quan nhất:"
  top_chunks.each_with_index do |chunk, i|
    puts "Ngữ cảnh #{i + 1}:"
    puts chunk[0..300] + "..."
    puts "=" * 50
  end

  # Bước 5: Tạo phản hồi
  puts "\n5. Tạo phản hồi AI..."
  system_prompt = "Bạn là một trợ lý AI chỉ trả lời dựa trên ngữ cảnh được cung cấp. Nếu câu trả lời không thể được suy ra trực tiếp từ ngữ cảnh được cung cấp, hãy trả lời: 'Tôi không có đủ thông tin để trả lời câu hỏi đó.'"

  user_prompt = top_chunks.map.with_index do |chunk, i|
    "Ngữ cảnh #{i + 1}:\n#{chunk}\n#{'=' * 50}\n"
  end.join("\n")
  user_prompt += "\nCâu hỏi: #{query}"

  ai_response = generate_response(system_prompt, user_prompt)

  if ai_response && ai_response['choices']
    puts "Phản hồi AI:"
    puts ai_response['choices'][0]['message']['content']
  else
    puts "Không thể tạo phản hồi AI"
    return
  end

  # Bước 6: Đánh giá phản hồi AI
  puts "\n6. Đánh giá phản hồi AI..."

  # Tải dữ liệu validation để lấy câu trả lời đúng
  validation_data = load_validation_data('data/val.json')

  if validation_data.empty?
    puts "Không có dữ liệu validation để đánh giá."
    puts "Tạo dữ liệu validation mẫu..."

    # Tạo dữ liệu mẫu nếu không có file validation
    sample_validation = [{
      'question' => query,
      'ideal_answer' => "Explainable AI (XAI) là một lĩnh vực nghiên cứu nhằm làm cho các hệ thống AI trở nên minh bạch và dễ hiểu hơn. XAI được coi là quan trọng vì nó giúp người dùng đánh giá tính công bằng và độ chính xác của các quyết định AI, xây dựng niềm tin, và cải thiện trách nhiệm giải trình."
    }]

    ideal_answer = sample_validation[0]['ideal_answer']
  else
    ideal_answer = validation_data[0]['ideal_answer'] || "Không có câu trả lời mẫu"
  end

  # Định nghĩa system prompt cho hệ thống đánh giá
  evaluate_system_prompt = "Bạn là một hệ thống đánh giá thông minh được giao nhiệm vụ đánh giá phản hồi của trợ lý AI. Nếu phản hồi của trợ lý AI rất gần với phản hồi đúng, hãy cho điểm 1. Nếu phản hồi không chính xác hoặc không thỏa mãn so với phản hồi đúng, hãy cho điểm 0. Nếu phản hồi một phần phù hợp với phản hồi đúng, hãy cho điểm 0.5."

  # Tạo prompt đánh giá bằng cách kết hợp user query, AI response, true response và evaluation system prompt
  evaluation_prompt = "Câu hỏi của người dùng: #{query}\nPhản hồi AI:\n#{ai_response['choices'][0]['message']['content']}\nPhản hồi đúng: #{ideal_answer}\n\nHãy đánh giá phản hồi của AI và đưa ra điểm số cùng với lý do."

  # Tạo phản hồi đánh giá
  evaluation_response = generate_response(evaluate_system_prompt, evaluation_prompt)

  if evaluation_response && evaluation_response['choices']
    puts "\nKết quả đánh giá:"
    puts evaluation_response['choices'][0]['message']['content']
  else
    puts "Không thể tạo đánh giá tự động."
  end
end

# ===============================================================================
# Các Hàm Tiện ích
# ===============================================================================

def load_validation_data(file_path)
  # Tải dữ liệu validation từ file JSON.
  #
  # Args:
  #   file_path (String): Đường dẫn đến file JSON.
  #
  # Returns:
  #   Array: Dữ liệu JSON đã được parse hoặc mảng rỗng nếu file không tồn tại.

  if File.exist?(file_path)
    JSON.parse(File.read(file_path))
  else
    puts "Không tìm thấy file validation: #{file_path}"
    []
  end
end

def evaluate_response(query, ai_response, true_response)
  # Đánh giá phản hồi AI so với phản hồi đúng.
  #
  # Args:
  #   query (String): Câu hỏi gốc
  #   ai_response (String): Phản hồi của AI
  #   true_response (String): Phản hồi mong đợi/đúng
  #
  # Returns:
  #   Float: Điểm đánh giá (0.0 đến 1.0)
  # Đánh giá đơn giản dựa trên keyword overlap
  # Trong production, bạn có thể sử dụng metrics tinh vi hơn
  ai_words = ai_response.downcase.split(/\W+/).reject(&:empty?)
  true_words = true_response.downcase.split(/\W+/).reject(&:empty?)

  common_words = ai_words & true_words
  total_words = (ai_words + true_words).uniq.length

  return 0.0 if total_words == 0

  common_words.length.to_f / total_words
end

# ===============================================================================
# Main Execution
# ===============================================================================

if __FILE__ == $0
  puts 'Triển khai Simple RAG bằng Ruby'
  puts '=' * 50

  run_simple_rag_demo
end

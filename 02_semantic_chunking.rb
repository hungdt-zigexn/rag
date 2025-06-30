#!/usr/bin/env ruby

# Giới thiệu về Semantic Chunking
# ================================
#
# Text chunking là một bước thiết yếu trong Retrieval-Augmented Generation (RAG),
# nơi các văn bản lớn được chia thành các đoạn có nghĩa để cải thiện độ chính xác của việc truy xuất.
# Khác với chunking có độ dài cố định, semantic chunking chia văn bản dựa trên độ tương tự
# nội dung giữa các câu.
#
# Các phương pháp Breakpoint:
# - Percentile: Tìm phần trăm thứ X của tất cả sự khác biệt về độ tương tự và chia chunks
#   khi độ giảm lớn hơn giá trị này.
# - Standard Deviation: Chia khi độ tương tự giảm nhiều hơn X độ lệch chuẩn so với trung bình.
# - Interquartile Range (IQR): Sử dụng khoảng tứ phân vị (Q3 - Q1) để xác định điểm chia.

require 'net/http'
require 'uri'
require 'json'
require 'matrix'
require 'pdf-reader'

# Đọc file .env nếu tồn tại
if File.exist?('.env')
  File.readlines('.env').each do |line|
    key, value = line.strip.split('=', 2)
    ENV[key] = value if key && value
  end
end

# Cấu hình API
API_BASE_URL = "https://api.studio.nebius.com/v1/"
API_KEY = ENV['OPENAI_API_KEY']

if API_KEY.nil? || API_KEY.empty?
  puts "Lỗi: Vui lòng đặt biến môi trường OPENAI_API_KEY"
  exit 1
else
  puts "API key đã được tải thành công"
end

puts "=== SEMANTIC CHUNKING RAG SYSTEM ==="
puts "Khởi tạo hệ thống semantic chunking..."

# Hàm trích xuất văn bản từ PDF
def extract_text_from_pdf(pdf_path)
  # Kiểm tra file PDF có tồn tại không
  if File.exist?(pdf_path)
    begin

      # Mở file PDF
      reader = PDF::Reader.new(pdf_path)
      all_text = ""

      # Lặp qua từng trang trong PDF
      reader.pages.each do |page|
        # Trích xuất text từ trang và thêm khoảng trắng
        all_text += page.text + " "
      end

      # Trả về text đã trích xuất, loại bỏ khoảng trắng đầu/cuối
      all_text.strip
    end
  end
end

# Cài đặt OpenAI API Client
def make_api_request(endpoint, payload)
  uri = URI("#{API_BASE_URL}#{endpoint}")
  http = Net::HTTP.new(uri.host, uri.port)
  http.use_ssl = true

  request = Net::HTTP::Post.new(uri)
  request['Content-Type'] = 'application/json'
  request['Authorization'] = "Bearer #{API_KEY}"
  request.body = payload.to_json

  response = http.request(request)

  if response.code.to_i >= 400
    puts "Lỗi API: #{response.code} - #{response.body}"
    return nil
  end

  JSON.parse(response.body)
end

# Tạo embedding cho văn bản
def get_embedding(text, model = "BAAI/bge-en-icl")
  payload = {
    model: model,
    input: text
  }

  response = make_api_request("embeddings", payload)
  return nil unless response && response['data'] && !response['data'].empty?

  response['data'][0]['embedding']
end

# Tính toán cosine similarity giữa hai vector
def cosine_similarity(vec1, vec2)
  # Chuyển đổi arrays thành vectors
  v1 = Vector[*vec1]
  v2 = Vector[*vec2]

  # Tính dot product
  dot_product = v1.inner_product(v2)

  # Tính magnitude (norm) của mỗi vector
  magnitude1 = Math.sqrt(v1.inner_product(v1))
  magnitude2 = Math.sqrt(v2.inner_product(v2))

  # Tránh chia cho 0
  return 0.0 if magnitude1 == 0.0 || magnitude2 == 0.0

  dot_product / (magnitude1 * magnitude2)
end

# Tính toán breakpoints dựa trên phương pháp được chọn
def compute_breakpoints(similarities, method = "percentile", threshold = 90)
  case method
  when "percentile"
    # Tính phần trăm thứ X của điểm tương tự
    sorted_sims = similarities.sort
    index = (threshold / 100.0 * (sorted_sims.length - 1)).round
    threshold_value = sorted_sims[index]
  when "standard_deviation"
    # Tính mean và standard deviation
    mean = similarities.sum.to_f / similarities.length
    variance = similarities.map { |x| (x - mean) ** 2 }.sum / similarities.length
    std_dev = Math.sqrt(variance)
    threshold_value = mean - (threshold * std_dev)
  when "interquartile"
    # Tính Q1 và Q3
    sorted_sims = similarities.sort
    q1_index = (0.25 * (sorted_sims.length - 1)).round
    q3_index = (0.75 * (sorted_sims.length - 1)).round
    q1 = sorted_sims[q1_index]
    q3 = sorted_sims[q3_index]
    threshold_value = q1 - 1.5 * (q3 - q1)
  else
    raise "Phương pháp không hợp lệ. Chọn 'percentile', 'standard_deviation', hoặc 'interquartile'."
  end

  # Tìm các chỉ số nơi độ tương tự giảm dưới ngưỡng
  breakpoints = []
  similarities.each_with_index do |sim, index|
    breakpoints << index if sim < threshold_value
  end

  breakpoints
end

# Chia văn bản thành các chunks semantic
def split_into_chunks(sentences, breakpoints)
  chunks = []
  start = 0

  # Lặp qua mỗi breakpoint để tạo chunks
  breakpoints.each do |bp|
    chunk_sentences = sentences[start..bp]
    chunks << chunk_sentences.join(". ") + "."
    start = bp + 1
  end

  # Thêm các câu còn lại như chunk cuối cùng
  if start < sentences.length
    remaining_sentences = sentences[start..-1]
    chunks << remaining_sentences.join(". ") unless remaining_sentences.empty?
  end

  chunks
end

# Tạo embeddings cho các text chunks
def create_embeddings(text_chunks)
  text_chunks.map { |chunk| get_embedding(chunk) }.compact
end

# Thực hiện semantic search
def semantic_search(query, text_chunks, chunk_embeddings, k = 5)
  # Tạo embedding cho query
  query_embedding = get_embedding(query)
  return [] unless query_embedding

  # Tính cosine similarity giữa query embedding và mỗi chunk embedding
  similarities = chunk_embeddings.map do |chunk_emb|
    cosine_similarity(query_embedding, chunk_emb)
  end

  # Lấy chỉ số của top-k chunks tương tự nhất
  indexed_similarities = similarities.each_with_index.to_a
  top_indices = indexed_similarities.sort_by { |sim, _| -sim }.first(k).map { |_, index| index }

  # Trả về top-k chunks liên quan nhất
  top_indices.map { |i| text_chunks[i] }
end

# Tạo phản hồi dựa trên retrieved chunks
def generate_response(system_prompt, user_message, model = "meta-llama/Llama-3.2-3B-Instruct")
  payload = {
    model: model,
    temperature: 0,
    messages: [
      { role: "system", content: system_prompt },
      { role: "user", content: user_message }
    ]
  }

  response = make_api_request("chat/completions", payload)
  return nil unless response && response['choices'] && !response['choices'].empty?

  response['choices'][0]['message']['content']
end

# === BƯỚC 1: TRÍCH XUẤT VĂN BẢN TỪ PDF ===
puts "\nBước 1: Trích xuất văn bản từ PDF..."
pdf_path = "data/AI_Information.pdf"
extracted_text = extract_text_from_pdf(pdf_path)
puts "Đã trích xuất #{extracted_text.length} ký tự từ PDF"
puts "500 ký tự đầu tiên:"
puts extracted_text[0..499]

# === BƯỚC 2: TẠO SENTENCE-LEVEL EMBEDDINGS ===
puts "\nBước 2: Tạo embeddings cấp độ câu..."
# Chia văn bản thành các câu (phân tách cơ bản)
sentences = extracted_text.split(". ")
puts "Đã chia thành #{sentences.length} câu"

# Tạo embeddings cho mỗi câu
puts "Đang tạo embeddings cho các câu..."
embeddings = []
sentences.each_with_index do |sentence, index|
  embedding = get_embedding(sentence)
  if embedding
    embeddings << embedding
  else
    puts "Không thể tạo embedding cho câu #{index + 1}"
  end
end

puts "Đã tạo #{embeddings.length} sentence embeddings"

# === BƯỚC 3: TÍNH TOÁN SỰ KHÁC BIỆT VỀ ĐỘ TƯƠNG TỰ ===
puts "\nBước 3: Tính toán độ tương tự giữa các câu liên tiếp..."
similarities = []
(0...embeddings.length - 1).each do |i|
  sim = cosine_similarity(embeddings[i], embeddings[i + 1])
  similarities << sim
end

puts "Đã tính toán #{similarities.length} điểm tương tự"

# === BƯỚC 4: THỰC HIỆN SEMANTIC CHUNKING ===
puts "\nBước 4: Thực hiện semantic chunking..."
# Tính toán breakpoints sử dụng phương pháp percentile với ngưỡng 90
breakpoints = compute_breakpoints(similarities, "percentile", 90)
puts "Đã tìm thấy #{breakpoints.length} breakpoints"

# Chia thành chunks sử dụng hàm split_into_chunks
text_chunks = split_into_chunks(sentences, breakpoints)
puts "Số lượng semantic chunks: #{text_chunks.length}"

puts "\nChunk đầu tiên:"
puts text_chunks[0]

# === BƯỚC 5: TẠO EMBEDDINGS CHO SEMANTIC CHUNKS ===
puts "\nBước 5: Tạo embeddings cho semantic chunks..."
chunk_embeddings = create_embeddings(text_chunks)
puts "Đã tạo #{chunk_embeddings.length} chunk embeddings"

# === BƯỚC 6: THỰC HIỆN SEMANTIC SEARCH ===
puts "\nBước 6: Thực hiện semantic search..."

# Tải dữ liệu validation từ file JSON
begin
  validation_data = JSON.parse(File.read('data/val.json'))
  query = validation_data[0]['question']
  ideal_answer = validation_data[0]['ideal_answer']
rescue => e
  puts "Không thể đọc file validation: #{e.message}"
  # Sử dụng query mẫu
  query = "What is 'Explainable AI' and why is it considered important?"
  ideal_answer = "Explainable AI (XAI) refers to artificial intelligence systems that can provide clear, understandable explanations for their decisions and actions. It is considered important because it helps build trust in AI systems by providing insights into how they make decisions, enables users to assess the fairness and accuracy of AI outputs, and supports accountability and transparency in AI applications."
end

# Lấy top 2 chunks liên quan
top_chunks = semantic_search(query, text_chunks, chunk_embeddings, 2)

puts "Query: #{query}"
top_chunks.each_with_index do |chunk, index|
  puts "\nContext #{index + 1}:"
  puts chunk
  puts "=" * 40
end

# === BƯỚC 7: TẠO PHẢN HỒI DỰA TRÊN RETRIEVED CHUNKS ===
puts "\nBước 7: Tạo phản hồi AI..."

# Định nghĩa system prompt cho AI assistant
system_prompt = "Bạn là một AI assistant chỉ trả lời dựa trên ngữ cảnh được cung cấp. Nếu câu trả lời không thể được suy ra trực tiếp từ ngữ cảnh được cung cấp, hãy trả lời: 'Tôi không có đủ thông tin để trả lời câu hỏi đó.'"

# Tạo user prompt dựa trên top chunks
user_prompt = ""
top_chunks.each_with_index do |chunk, index|
  user_prompt += "Context #{index + 1}:\n#{chunk}\n#{'=' * 35}\n\n"
end
user_prompt += "Question: #{query}"

# Tạo phản hồi AI
ai_response = generate_response(system_prompt, user_prompt)

if ai_response
  puts "\nPhản hồi AI:"
  puts ai_response
else
  puts "Không thể tạo phản hồi AI"
end

# === BƯỚC 8: ĐÁNH GIÁ PHẢN HỒI AI ===
puts "\nBước 8: Đánh giá phản hồi AI..."

if ai_response
  # Định nghĩa system prompt cho hệ thống đánh giá
  evaluate_system_prompt = "Bạn là một hệ thống đánh giá thông minh được giao nhiệm vụ đánh giá phản hồi của AI assistant. Nếu phản hồi của AI assistant rất gần với phản hồi thực, hãy gán điểm 1. Nếu phản hồi không chính xác hoặc không thỏa mãn so với phản hồi thực, hãy gán điểm 0. Nếu phản hồi một phần phù hợp với phản hồi thực, hãy gán điểm 0.5."

  # Tạo evaluation prompt
  evaluation_prompt = "User Query: #{query}\nAI Response:\n#{ai_response}\nTrue Response: #{ideal_answer}\n#{evaluate_system_prompt}"

  # Tạo phản hồi đánh giá
  evaluation_response = generate_response(evaluate_system_prompt, evaluation_prompt)

  if evaluation_response
    puts "\nKết quả đánh giá:"
    puts evaluation_response
  else
    puts "Không thể tạo đánh giá"
  end
end

puts "\n=== HOÀN THÀNH SEMANTIC CHUNKING RAG ==="

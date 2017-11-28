#!/usr/bin/env ruby
`rm -f final.py`
%w{read_student_vectors.py}.each do |file|
  `cat #{file} >> 'final.py'`
end

puts `python final.py`

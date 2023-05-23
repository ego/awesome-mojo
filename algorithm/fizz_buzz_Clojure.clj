;; https://onecompiler.com/clojure/3z9d5g3uf
(import '[java.io OutputStream])
(require '[clojure.java.io :as io])

(def devnull (io/writer (OutputStream/nullOutputStream)))


(defmacro timeit [n expr]
  `(with-out-str (time
                   (dotimes [_# ~(Math/pow 1 n)]
                     (binding [*out* devnull]
                       ~expr)))))


(defn fizz-buzz [n]
  (print
    (apply str
      (for [i (range 1 (inc n))]
        (cond
          (zero? (mod i 15)) "FizzBuzz\n" 
          (zero? (mod i 5))  "Buzz\n"
          (zero? (mod i 3))  "Fizz\n"
          :else              (str i "\n"))))))


(print (timeit 100 (fizz-buzz 100)))

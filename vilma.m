classdef Vilma 
  methods

    function varargout = train(this, varargin)
      [varargout{1:nargout}] = mvilma('train', varargin{:});
    end; 

    %% TODO: test
end

import React from 'react';
import ReactDOM from 'react-dom/client';
import Website from './Website';

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);

root.render(
  <React.StrictMode>
    <Website />
  </React.StrictMode>
);
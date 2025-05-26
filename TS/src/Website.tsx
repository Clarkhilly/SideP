import React from 'react';

const Website: React.FC = () => {
  return (
    <div style={{ fontFamily: 'Arial, sans-serif', margin: 0, padding: '20px' }}>
      <header style={{ textAlign: 'center', marginBottom: '40px' }}>
        <h1>Welcome to My Personal Website</h1>
        <nav>
          <a href="#about" style={{ margin: '0 15px', textDecoration: 'none' }}>About</a>
          <a href="#projects" style={{ margin: '0 15px', textDecoration: 'none' }}>Projects</a>
          <a href="#contact" style={{ margin: '0 15px', textDecoration: 'none' }}>Contact</a>
        </nav>
      </header>

      <main>
        <section id="about" style={{ marginBottom: '40px' }}>
          <h2>About Me</h2>
          <p>Hello! I'm a developer passionate about creating awesome web experiences.</p>
        </section>

        <section id="projects" style={{ marginBottom: '40px' }}>
          <h2>My Projects</h2>
          <div>
            <h3>Project 1</h3>
            <p>Description of your first project.</p>
          </div>
        </section>

        <section id="contact">
          <h2>Contact</h2>
          <p>Feel free to reach out to me!</p>
        </section>
      </main>
    </div>
  );
};

export default Website;
